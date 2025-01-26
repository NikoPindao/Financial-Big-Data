import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

from src.data_processing.data_merger import CryptoDataMerger
from src.clustering.regime_detection.short_term import ShortTermDetector

class LSTMRegimePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, num_regimes: int = 5):
        super(LSTMRegimePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_regimes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM layers
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output with attention
        out = self.fc1(attn_out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

class RegimePredictionTester:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.regime_mapping = {
            'stable_bull': 0,
            'volatile_bull': 1,
            'stable_bear': 2,
            'volatile_bear': 3,
            'consolidation': 4
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create base directory for plots
        self.plots_dir = Path("data/plots/lstm_analysis")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {self.plots_dir.absolute()}")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare enhanced feature set for LSTM"""
        features = {}
        
        returns = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        log_returns = np.log(df['close']).diff().replace([np.inf, -np.inf], np.nan)
        
        #Features we use for the LSTM
        features.update({
            'returns': returns,
            'log_returns': log_returns,
            'volatility_5h': returns.rolling(window=5, min_periods=1).std(),
            'volatility_24h': returns.rolling(window=24, min_periods=1).std(),
            'trend_5h': returns.rolling(window=5, min_periods=1).mean(),
            'trend_24h': returns.rolling(window=24, min_periods=1).mean(),
            'price_range': ((df['high'] - df['low']) / df['close']).clip(-10, 10) 
        })
        
        #We clip extreme values for features
        volume_ma = df['volume'].rolling(window=24, min_periods=1).mean()
        volume_ma = volume_ma.replace(0, df['volume'].mean()) 
        volume_change = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
        volume_std = df['volume'].rolling(window=24, min_periods=1).std()
        
        features.update({
            'volume_intensity': (df['volume'] / volume_ma).clip(0, 10),
            'volume_change': volume_change.clip(-10, 10),
            'volume_volatility': (volume_std / volume_ma).clip(0, 10)
        })
        
        rsi = self.calculate_rsi(df['close'])
        features['rsi'] = rsi.clip(0, 100)  # RSI should be between 0 and 100
        
        # MACD
        macd, signal, hist = self.calculate_macd(df['close'])
        max_macd = max(abs(macd.max()), abs(macd.min()))
        max_signal = max(abs(signal.max()), abs(signal.min()))
        max_hist = max(abs(hist.max()), abs(hist.min()))
        
        features.update({
            'macd': (macd / max_macd if max_macd != 0 else macd).clip(-1, 1),
            'macd_signal': (signal / max_signal if max_signal != 0 else signal).clip(-1, 1),
            'macd_hist': (hist / max_hist if max_hist != 0 else hist).clip(-1, 1)
        })
        
        # Bollinger Bands
        bb_position, bb_width = self.calculate_bollinger_bands(df['close'])
        features.update({
            'bb_position': bb_position.clip(0, 1),  # Should be between 0 and 1
            'bb_width': bb_width.clip(0, 5)  # Clip extreme width values
        })
        
        features_df = pd.DataFrame(features)
        features_df = features_df.ffill().bfill()
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        features_df = features_df.astype(float)
        
        #Prevent non-finite values
        if not np.all(np.isfinite(features_df)):
            raise ValueError("Features contain non-finite values after processing")
        
        try:
            scaled_features = self.scaler.fit_transform(features_df)
            if not np.all(np.isfinite(scaled_features)):
                raise ValueError("Scaled features contain non-finite values")
            
            return scaled_features
            
        except Exception as e:
            print(f"Error during feature scaling: {str(e)}")
            print("Feature statistics before scaling:")
            print(features_df.describe())
            raise
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator with safety checks"""
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS with safety check for division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI value
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicators with safety checks"""
        exp1 = prices.ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = prices.ewm(span=26, adjust=False, min_periods=1).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
        histogram = macd - signal
        
        # Replace any infinities or NaNs
        macd = macd.replace([np.inf, -np.inf], np.nan).fillna(0)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0)
        histogram = histogram.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return macd, signal, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands with safety checks"""
        ma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        
        upper = ma + (std * 2)
        lower = ma - (std * 2)
        
        # Calculate position with safety checks
        band_width = upper - lower
        position = (prices - lower) / band_width.replace(0, np.nan)
        width = std / ma.replace(0, np.nan)
        
        # Clean up any infinities or NaNs
        position = position.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        width = width.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return position, width
    
    def create_sequences(self, features: np.ndarray, regimes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(regimes[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train_and_evaluate(self, data: pd.DataFrame, regimes: pd.DataFrame, symbol: str):
        """Train and evaluate LSTM model for a single symbol"""
        print(f"\nPreparing features for {symbol}...")
        try:
            # Prepare features and targets
            features = self.prepare_features(data)
            print(f"Generated {features.shape[1]} features")
            
            regime_values = regimes['regime'].map(self.regime_mapping).values
            print(f"Regime distribution: {pd.Series(regime_values).value_counts().to_dict()}")
            
            # Create sequences
            X, y = self.create_sequences(features, regime_values)
            print(f"Created {len(X)} sequences of length {self.sequence_length}")
            
            # Split data
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            X_train = torch.FloatTensor(X[:train_size]).to(self.device)
            y_train = torch.LongTensor(y[:train_size]).to(self.device)
            X_val = torch.FloatTensor(X[train_size:train_size+val_size]).to(self.device)
            y_val = torch.LongTensor(y[train_size:train_size+val_size]).to(self.device)
            X_test = torch.FloatTensor(X[train_size+val_size:]).to(self.device)
            y_test = torch.LongTensor(y[train_size+val_size:]).to(self.device)
            
            print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Create model
            input_size = features.shape[1]
            model = LSTMRegimePredictor(input_size=input_size).to(self.device)
            print(f"Created LSTM model with input size {input_size}")
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training
            num_epochs = 5  #You can do more than 5 periods
            batch_size = 32
            train_losses = []
            val_accuracies = []
            best_val_accuracy = 0
            best_model = None
            
            print("\nStarting training...")
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                batch_count = 0
                
                # Training loop with progress logging
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Log progress every 10 batches
                    if batch_count % 10 == 0:
                        print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_count}/{len(X_train)//batch_size} - Loss: {loss.item():.4f}")
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    _, predicted = torch.max(val_outputs, 1)
                    accuracy = accuracy_score(y_val.cpu(), predicted.cpu())
                    
                    # Save best model
                    if accuracy > best_val_accuracy:
                        best_val_accuracy = accuracy
                        best_model = model.state_dict()
                
                avg_loss = total_loss / (len(X_train) // batch_size)
                train_losses.append(avg_loss)
                val_accuracies.append(accuracy)
                
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Average Loss: {avg_loss:.4f}')
                print(f'Validation Accuracy: {accuracy:.4f}')
                print('-' * 50)
            
            # Load best model for testing
            if best_model is not None:
                model.load_state_dict(best_model)
                print(f"Using best model with validation accuracy: {best_val_accuracy:.4f}")
            
            # Test evaluation
            print("\nEvaluating on test set...")
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs, 1)
                test_accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
                test_f1 = f1_score(y_test.cpu(), predicted.cpu(), average='weighted')
                
                # Calculate per-class metrics
                class_f1 = f1_score(y_test.cpu(), predicted.cpu(), average=None)
                
                print("\nPer-class F1 scores:")
                for regime, idx in self.regime_mapping.items():
                    print(f"{regime}: {class_f1[idx]:.4f}")
            
            # Create visualizations
            print("\nCreating visualizations...")
            self.plot_training_history(train_losses, val_accuracies, symbol)
            self.plot_confusion_matrix(y_test.cpu(), predicted.cpu(), symbol)
            
            return {
                'accuracy': test_accuracy,
                'f1_score': test_f1,
                'per_class_f1': dict(zip(self.regime_mapping.keys(), class_f1)),
                'model': model,
                'training_history': {
                    'losses': train_losses,
                    'val_accuracies': val_accuracies
                }
            }
            
        except Exception as e:
            print(f"Error in training process: {str(e)}")
            traceback.print_exc()
            return None
    
    def plot_training_history(self, train_losses: List[float], val_accuracies: List[float], symbol: str):
        """Plot training history"""
        # Sanitize symbol name for file path
        safe_symbol = symbol.replace('/', '_')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(y=train_losses, name='Training Loss', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=val_accuracies, name='Validation Accuracy', 
                               line=dict(color='red'), yaxis='y2'))
        
        fig.update_layout(
            title=f'Training History - {symbol}',
            yaxis=dict(title='Loss', side='left'),
            yaxis2=dict(title='Accuracy', side='right', overlaying='y'),
            height=600,
            showlegend=True
        )
        
        # Create subdirectory for each symbol
        symbol_dir = self.plots_dir / safe_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        fig.write_html(symbol_dir / "training_history.html")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, symbol: str):
        """Plot confusion matrix"""
        # Sanitize symbol name for file path
        safe_symbol = symbol.replace('/', '_')
        
        cm = confusion_matrix(y_true, y_pred)
        labels = list(self.regime_mapping.keys())
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {symbol}',
            xaxis_title='Predicted Regime',
            yaxis_title='True Regime',
            height=800,
            width=800
        )
        
        # Create subdirectory for each symbol
        symbol_dir = self.plots_dir / safe_symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        fig.write_html(symbol_dir / "confusion_matrix.html")

def main():
    # Initialize components
    print("Initializing components...")
    merger = CryptoDataMerger()
    regime_detector = ShortTermDetector()
    tester = RegimePredictionTester()
    
    # Load data
    print("\nLoading data...")
    try:
        data = merger.merge_crypto_data()
        print(f"Loaded data with shape: {data.shape}")
        
        # Test for a few major cryptocurrencies
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        results = {}
        
        for symbol in test_symbols:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}")
            print('='*50)
            
            try:
                # Get symbol data and ensure it's not empty
                symbol_data = data.xs(symbol, level='symbol')
                print(f"Symbol data shape: {symbol_data.shape}")
                
                if symbol_data.empty:
                    print(f"No data available for {symbol}")
                    continue
                
                # Detect regimes
                print("\nDetecting regimes...")
                regimes, _ = regime_detector.detect_regime(symbol_data)
                if regimes.empty:
                    print(f"No regimes detected for {symbol}")
                    continue
                
                print("Regime distribution:")
                print(regimes['regime'].value_counts())
                
                # Train and evaluate
                print("\nStarting LSTM training and evaluation...")
                symbol_results = tester.train_and_evaluate(symbol_data, regimes, symbol)
                
                if symbol_results is not None:
                    results[symbol] = symbol_results
                    print("\nResults:")
                    print(f"Test Accuracy: {symbol_results['accuracy']:.4f}")
                    print(f"Overall F1 Score: {symbol_results['f1_score']:.4f}")
                    print("\nPer-class F1 scores:")
                    for regime, score in symbol_results['per_class_f1'].items():
                        print(f"{regime}: {score:.4f}")
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                traceback.print_exc()
        
        # Print overall results
        if results:
            print("\n" + "="*50)
            print("Overall Results Summary")
            print("="*50)
            
            for symbol, res in results.items():
                print(f"\n{symbol}:")
                print(f"Accuracy: {res['accuracy']:.4f}")
                print(f"F1 Score: {res['f1_score']:.4f}")
                print("Per-class F1 scores:")
                for regime, score in res['per_class_f1'].items():
                    print(f"  {regime}: {score:.4f}")
        else:
            print("\nNo results were generated. Check the errors above.")
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 