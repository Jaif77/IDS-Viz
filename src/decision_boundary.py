# src/decision_boundary.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class DecisionBoundaryExplorer:
    """Creates 2D visualization of model decision boundaries"""
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
    
    def reduce_dimensions(self, X, method='pca', n_components=2, perplexity=30):
        """Reduce high-dimensional data to 2D using PCA or t-SNE"""
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X)
            method_name = 'PCA'
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            X_reduced = reducer.fit_transform(X)
            method_name = 't-SNE'
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        return X_reduced, method_name
    
    def create_scatter_plot(self, X_2d, y_true, y_pred, y_proba=None, class_names=None, method_name='PCA'):
        """Create interactive scatter plot with misclassifications highlighted"""
        
        # Get confidence scores if probabilities are provided
        if y_proba is not None:
            if len(y_proba.shape) == 2:
                # Multi-class: take max probability as confidence
                confidence = np.max(y_proba, axis=1)
            else:
                # Binary: probability of positive class
                confidence = y_proba
        else:
            confidence = np.ones(len(y_true))  # Default if not available
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        plot_df['x'] = X_2d[:, 0]
        plot_df['y'] = X_2d[:, 1]
        plot_df['True Label'] = y_true
        plot_df['Predicted'] = y_pred
        plot_df['Correct'] = (y_true == y_pred)
        plot_df['Confidence'] = confidence
        
        # Convert numeric labels to names if provided
        if class_names is not None:
            plot_df['True Label Name'] = plot_df['True Label'].map(
                lambda x: class_names[x] if isinstance(x, (int, np.integer)) else x
            )
            plot_df['Predicted Name'] = plot_df['Predicted'].map(
                lambda x: class_names[x] if isinstance(x, (int, np.integer)) else x
            )
        else:
            plot_df['True Label Name'] = plot_df['True Label'].astype(str)
            plot_df['Predicted Name'] = plot_df['Predicted'].astype(str)
        
        fig = go.Figure()
        
        # ========== 1. CORRECT PREDICTIONS (Colored by confidence) ==========
        correct_df = plot_df[plot_df['Correct'] == True]
        
        # Separate Benign and Attack for better coloring
        benign_correct = correct_df[correct_df['True Label Name'].str.upper().str.contains('BENIGN', na=False)]
        attack_correct = correct_df[~correct_df['True Label Name'].str.upper().str.contains('BENIGN', na=False)]
        
        # Benign points - green scale based on confidence
        if len(benign_correct) > 0:
            fig.add_trace(go.Scatter(
                x=benign_correct['x'], y=benign_correct['y'],
                mode='markers',
                name='Benign (Correct)',
                marker=dict(
                    size=8,
                    color=benign_correct['Confidence'],
                    colorscale='Greens',
                    colorbar=dict(title="Confidence", x=1.02),
                    opacity=0.7,
                    showscale=False
                ),
                hovertemplate='<b>✅ CORRECT</b><br>' +
                              '<b>True:</b> %{customdata[0]}<br>' +
                              '<b>Predicted:</b> %{customdata[1]}<br>' +
                              '<b>Confidence:</b> %{customdata[2]:.2%}<br>' +
                              '<b>Type:</b> Benign<extra></extra>',
                customdata=benign_correct[['True Label Name', 'Predicted Name', 'Confidence']].values
            ))
        
        # Attack points - red scale based on confidence
        if len(attack_correct) > 0:
            fig.add_trace(go.Scatter(
                x=attack_correct['x'], y=attack_correct['y'],
                mode='markers',
                name='Attack (Correct)',
                marker=dict(
                    size=8,
                    color=attack_correct['Confidence'],
                    colorscale='Reds',
                    opacity=0.7,
                    showscale=False
                ),
                hovertemplate='<b>✅ CORRECT</b><br>' +
                              '<b>True:</b> %{customdata[0]}<br>' +
                              '<b>Predicted:</b> %{customdata[1]}<br>' +
                              '<b>Confidence:</b> %{customdata[2]:.2%}<br>' +
                              '<b>Type:</b> Attack<extra></extra>',
                customdata=attack_correct[['True Label Name', 'Predicted Name', 'Confidence']].values
            ))
        
        # ========== 2. MISCLASSIFICATIONS (Yellow X markers with details) ==========
        wrong_df = plot_df[plot_df['Correct'] == False]
        
        if len(wrong_df) > 0:
            fig.add_trace(go.Scatter(
                x=wrong_df['x'], y=wrong_df['y'],
                mode='markers',
                name='⚠️ Misclassified',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='#F1C40F',
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>⚠️ MISCLASSIFIED!</b><br>' +
                              '<b>True:</b> %{customdata[0]}<br>' +
                              '<b>Predicted:</b> %{customdata[1]}<br>' +
                              '<b>Confidence:</b> %{customdata[2]:.2%}<extra></extra>',
                customdata=wrong_df[['True Label Name', 'Predicted Name', 'Confidence']].values
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Decision Boundary Visualization ({method_name})',
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            hovermode='closest',
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            height=650,
            clickmode='event+select'
        )
        
        return fig
    
    def get_statistics(self, X_2d, y_true, y_pred, y_proba=None):
        """Calculate statistics about the visualization"""
        
        total = len(y_true)
        correct = sum(y_true == y_pred)
        wrong = total - correct
        
        from scipy.spatial import distance
        if len(X_2d) > 1:
            avg_distance = np.mean([distance.euclidean(X_2d[i], X_2d[j]) 
                                   for i in range(min(100, len(X_2d))) 
                                   for j in range(i+1, min(100, len(X_2d)))])
        else:
            avg_distance = 0
        
        stats = {
            'total_samples': total,
            'correct_count': correct,
            'wrong_count': wrong,
            'accuracy': correct / total if total > 0 else 0,
            'avg_point_distance': round(avg_distance, 3)
        }
        
        # Add class-wise statistics if probabilities available
        if y_proba is not None:
            stats['avg_confidence'] = np.mean(np.max(y_proba, axis=1) if len(y_proba.shape) == 2 else y_proba)
        
        return stats
    
    def get_misclassified_samples(self, X, y_true, y_pred, feature_names):
        """Return dataframe of all misclassified samples for export"""
        misclassified_mask = (y_true != y_pred)
        
        misclassified_df = pd.DataFrame(X[misclassified_mask], columns=feature_names)
        misclassified_df['True_Label'] = y_true[misclassified_mask]
        misclassified_df['Predicted_Label'] = y_pred[misclassified_mask]
        
        return misclassified_df
    
    def get_cluster_regions(self, X_2d, y_true, y_pred):
        """Identify regions where misclassifications cluster"""
        from scipy.spatial import distance
        
        misclassified_mask = (y_true != y_pred)
        
        if sum(misclassified_mask) < 3:
            return None
        
        # Get centroids of misclassified points
        misclassified_points = X_2d[misclassified_mask]
        centroid = np.mean(misclassified_points, axis=0)
        
        # Calculate average distance from centroid
        distances = [distance.euclidean(p, centroid) for p in misclassified_points]
        
        clusters = {
            'has_blind_spot': True,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'cluster_radius': np.mean(distances),
            'points_in_cluster': len(misclassified_points)
        }
        
        return clusters