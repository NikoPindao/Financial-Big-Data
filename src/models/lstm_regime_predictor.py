import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from typing import Tuple, Dict, List

class LSTMRegimePredictor(nn.Module):
    """LSTM model for regime prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_regimes: int):
        super(LSTMRegimePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_regimes)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

class RegimePredictionTrainer:
    """Class for training and evaluating regime prediction models"""
    
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
        self.inverse_regime_mapping = {v: k for k, v in self.regime_mapping.items()}
    
    def prepare_data(self, df: pd.DataFrame, regimes: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for LSTM"""
        # Prepare features
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:(i + self.sequence_length)])
            regime = regimes['regime'].iloc[i + self.sequence_length]
            y.append(self.regime_mapping[regime])
        
        return np.array(X), np.array(y)
    
    def train_model(self, train_data: Tuple[np.ndarray, np.ndarray], 
                   val_data: Tuple[np.ndarray, np.ndarray],
                   hidden_size: int = 64, num_layers: int = 2,
                   num_epochs: int = 50, batch_size: int = 32,
                   learning_rate: float = 0.001) -> Tuple[LSTMRegimePredictor, Dict]:
        """Train LSTM model"""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create model
        input_size = X_train.shape[2]
        num_regimes = len(self.regime_mapping)
        model = LSTMRegimePredictor(input_size, hidden_size, num_layers, num_regimes)
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.FloatTensor(X_val))
                _, predicted = torch.max(val_outputs, 1)
                accuracy = accuracy_score(y_val, predicted.numpy())
            
            train_losses.append(total_loss / len(train_loader))
            val_accuracies.append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {accuracy:.4f}')
        
        training_history = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
        
        return model, training_history
    
    def evaluate_model(self, model: LSTMRegimePredictor, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Evaluate model performance"""
        X_test, y_test = test_data
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(torch.FloatTensor(X_test))
            _, predicted = torch.max(test_outputs, 1)
            predicted = predicted.numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predicted)
        f1 = f1_score(y_test, predicted, average='weighted')
        conf_matrix = confusion_matrix(y_test, predicted)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'predictions': predicted
        }
    
    def plot_training_history(self, history: Dict, save_path: str = None):
        """Plot training history"""
        fig = go.Figure()
        
        # Plot training loss
        fig.add_trace(
            go.Scatter(
                y=history['train_losses'],
                name='Training Loss',
                line=dict(color='blue')
            )
        )
        
        # Plot validation accuracy
        fig.add_trace(
            go.Scatter(
                y=history['val_accuracies'],
                name='Validation Accuracy',
                line=dict(color='red'),
                yaxis='y2'
            )
        )
        
        fig.update_layout(
            title='Training History',
            yaxis=dict(title='Loss', side='left'),
            yaxis2=dict(title='Accuracy', side='right', overlaying='y'),
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: str = None):
        """Plot confusion matrix"""
        labels = list(self.regime_mapping.keys())
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Regime',
            yaxis_title='True Regime',
            height=800,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

def predict_regimes(data: pd.DataFrame, regimes: pd.DataFrame, save_dir: str):
    """Main function to train and evaluate regime prediction model"""
    trainer = RegimePredictionTrainer()
    
    # Prepare data
    X, y = trainer.prepare_data(data, regimes)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Train model
    model, history = trainer.train_model(
        (X_train, y_train),
        (X_val, y_val),
        hidden_size=128,
        num_layers=3,
        num_epochs=100
    )
    
    # Evaluate model
    results = trainer.evaluate_model(model, (X_test, y_test))
    
    # Create visualizations
    trainer.plot_training_history(history, f"{save_dir}/training_history.html")
    trainer.plot_confusion_matrix(results['confusion_matrix'], f"{save_dir}/confusion_matrix.html")
    
    print("\nModel Performance:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    return model, results 