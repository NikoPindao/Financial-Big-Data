import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from datetime import datetime

class ResultsVisualizer:
    def __init__(self):
        self.plots_dir = Path("data/plots/classification_results")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for regimes
        self.regime_colors = {
            'stable_bull': '#2ecc71',    # Green
            'volatile_bull': '#27ae60',   # Dark Green
            'stable_bear': '#e74c3c',     # Red
            'volatile_bear': '#c0392b',   # Dark Red
            'consolidation': '#3498db'    # Blue
        }
    
    def plot_regime_distribution_over_time(self, data: pd.DataFrame, symbol: str):
        """Plot regime distribution over time"""
        fig = go.Figure()
        
        # Calculate daily regime proportions
        daily_regimes = data.groupby([pd.Grouper(freq='D'), 'regime']).size().unstack(fill_value=0)
        daily_regimes_pct = daily_regimes.div(daily_regimes.sum(axis=1), axis=0)
        
        # Create stacked area plot
        for regime in daily_regimes_pct.columns:
            fig.add_trace(
                go.Scatter(
                    x=daily_regimes_pct.index,
                    y=daily_regimes_pct[regime],
                    name=regime,
                    stackgroup='one',
                    fillcolor=self.regime_colors[regime],
                    line=dict(color=self.regime_colors[regime])
                )
            )
        
        fig.update_layout(
            title=f"Regime Distribution Over Time - {symbol}",
            xaxis_title="Date",
            yaxis_title="Proportion",
            hovermode='x unified',
            showlegend=True
        )
        
        fig.write_html(self.plots_dir / f"{symbol}_regime_distribution.html")
    
    def plot_regime_transitions(self, transitions: pd.DataFrame, symbol: str):
        """Plot regime transition patterns"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Transition Frequencies',
                'Average Duration by Regime',
                'Transition Types',
                'Regime Stability'
            )
        )
        
        # Transition frequencies
        transition_counts = transitions.groupby(['from_regime', 'to_regime']).size().reset_index(name='count')
        
        fig.add_trace(
            go.Heatmap(
                x=transition_counts['to_regime'],
                y=transition_counts['from_regime'],
                z=transition_counts['count'],
                colorscale='Viridis'
            ),
            row=1, col=1
        )
        
        # Average duration by regime
        duration_by_regime = transitions.groupby('from_regime')['duration_days'].mean()
        
        fig.add_trace(
            go.Bar(
                x=duration_by_regime.index,
                y=duration_by_regime.values,
                marker_color=[self.regime_colors[r] for r in duration_by_regime.index]
            ),
            row=1, col=2
        )
        
        # Transition types
        if 'is_volatility_increase' in transitions.columns:
            type_counts = pd.DataFrame({
                'Type': ['Volatility Increase', 'Trend Change'],
                'Count': [
                    transitions['is_volatility_increase'].sum(),
                    transitions['is_trend_change'].sum()
                ]
            })
            
            fig.add_trace(
                go.Bar(
                    x=type_counts['Type'],
                    y=type_counts['Count']
                ),
                row=2, col=1
            )
        
        # Regime stability
        regime_changes = transitions.groupby('from_regime').size()
        total_periods = len(transitions)
        stability = 1 - (regime_changes / total_periods)
        
        fig.add_trace(
            go.Bar(
                x=stability.index,
                y=stability.values,
                marker_color=[self.regime_colors[r] for r in stability.index]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Regime Transition Analysis - {symbol}",
            showlegend=False
        )
        
        fig.write_html(self.plots_dir / f"{symbol}_transitions.html")
    
    def plot_prediction_performance(self, true_regimes: np.ndarray, predicted_regimes: np.ndarray, 
                                  symbol: str, history: dict = None):
        """Plot prediction performance metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Confusion Matrix',
                'Training History',
                'Prediction Distribution',
                'Accuracy by Regime'
            )
        )
        
        # Confusion Matrix
        cm = pd.crosstab(true_regimes, predicted_regimes)
        
        fig.add_trace(
            go.Heatmap(
                z=cm.values,
                x=cm.columns,
                y=cm.index,
                colorscale='Viridis',
                showscale=False
            ),
            row=1, col=1
        )
        
        # Training History
        if history:
            fig.add_trace(
                go.Scatter(y=history['losses'], name='Training Loss'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['val_accuracies'], name='Validation Accuracy'),
                row=1, col=2
            )
        
        # Prediction Distribution
        pred_dist = pd.Series(predicted_regimes).value_counts()
        true_dist = pd.Series(true_regimes).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=pred_dist.index,
                y=pred_dist.values,
                name='Predicted',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=true_dist.index,
                y=true_dist.values,
                name='True',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Accuracy by Regime
        regime_accuracy = {}
        for regime in np.unique(true_regimes):
            mask = true_regimes == regime
            regime_accuracy[regime] = np.mean(predicted_regimes[mask] == true_regimes[mask])
        
        fig.add_trace(
            go.Bar(
                x=list(regime_accuracy.keys()),
                y=list(regime_accuracy.values()),
                marker_color=[self.regime_colors[r] for r in regime_accuracy.keys()]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1000,
            title_text=f"Prediction Performance Analysis - {symbol}",
            showlegend=True
        )
        
        fig.write_html(self.plots_dir / f"{symbol}_prediction_performance.html")
    
    def create_summary_dashboard(self, results: dict):
        """Create a summary dashboard of all results"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Overall Accuracy by Symbol',
                'F1 Scores by Regime',
                'Prediction Distribution',
                'Training Convergence'
            )
        )
        
        # Overall Accuracy
        accuracies = {symbol: res['accuracy'] for symbol, res in results.items()}
        
        fig.add_trace(
            go.Bar(
                x=list(accuracies.keys()),
                y=list(accuracies.values()),
                name='Accuracy'
            ),
            row=1, col=1
        )
        
        # F1 Scores by Regime
        f1_scores = {}
        for symbol, res in results.items():
            for regime, score in res['per_class_f1'].items():
                if regime not in f1_scores:
                    f1_scores[regime] = []
                f1_scores[regime].append(score)
        
        for regime, scores in f1_scores.items():
            fig.add_trace(
                go.Box(
                    y=scores,
                    name=regime,
                    marker_color=self.regime_colors[regime]
                ),
                row=1, col=2
            )
        
        # Average Prediction Distribution
        all_predictions = []
        for res in results.values():
            if 'predictions' in res:
                all_predictions.extend(res['predictions'])
        
        if all_predictions:
            pred_dist = pd.Series(all_predictions).value_counts(normalize=True)
            
            fig.add_trace(
                go.Bar(
                    x=pred_dist.index,
                    y=pred_dist.values,
                    marker_color=[self.regime_colors[r] for r in pred_dist.index]
                ),
                row=2, col=1
            )
        
        # Training Convergence
        for symbol, res in results.items():
            if 'training_history' in res:
                fig.add_trace(
                    go.Scatter(
                        y=res['training_history']['val_accuracies'],
                        name=f"{symbol} Validation",
                        line=dict(dash='dash')
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=1000,
            title_text="Classification Results Summary",
            showlegend=True
        )
        
        fig.write_html(self.plots_dir / "summary_dashboard.html")

def main():
    # Example usage
    visualizer = ResultsVisualizer()
    
    # Load your results and create visualizations
    # This is just an example - replace with your actual data
    from test_lstm_prediction import main as run_lstm
    results = run_lstm()  # This should return the results dictionary
    
    if results:
        visualizer.create_summary_dashboard(results)
        
        for symbol, res in results.items():
            if 'predictions' in res and 'true_values' in res:
                visualizer.plot_prediction_performance(
                    res['true_values'],
                    res['predictions'],
                    symbol,
                    res.get('training_history')
                )

if __name__ == "__main__":
    main() 