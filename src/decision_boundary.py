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

        # Convert to numpy arrays
        X_2d = np.array(X_2d)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        plot_df['x'] = X_2d[:, 0]
        plot_df['y'] = X_2d[:, 1]
        plot_df['y_true'] = y_true
        plot_df['y_pred'] = y_pred
        plot_df['correct'] = (y_true == y_pred)

        # Separate correct and wrong
        correct_df = plot_df[plot_df['correct'] == True]
        wrong_df = plot_df[plot_df['correct'] == False]

        fig = go.Figure()

        # ========== CORRECT POINTS ==========
        if len(correct_df) > 0:
            fig.add_trace(go.Scatter(
                x=correct_df['x'].tolist(),
                y=correct_df['y'].tolist(),
                mode='markers',
                name='✅ Correct',
                marker=dict(
                    size=6,
                    color='#2ECC71',
                    opacity=0.7
                )
            ))

        # ========== MISCLASSIFIED POINTS ==========
        if len(wrong_df) > 0:
            fig.add_trace(go.Scatter(
                x=wrong_df['x'].tolist(),
                y=wrong_df['y'].tolist(),
                mode='markers',
                name='⚠️ Misclassified',
                marker=dict(
                    symbol='x',
                    size=10,
                    color='#F1C40F',
                    line=dict(width=2, color='black')
                )
            ))

        # ========== LAYOUT ==========
        fig.update_layout(
            title=f'Decision Boundary Visualization ({method_name})',
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            height=650,
            showlegend=True,
            plot_bgcolor='rgba(240, 240, 240, 0.1)'
        )

        return fig

    def get_statistics(self, X_2d, y_true, y_pred, y_proba=None):
        """Calculate statistics about the visualization"""

        total = len(y_true)
        correct = int(sum(y_true == y_pred))
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

        if y_proba is not None:
            stats['avg_confidence'] = float(np.mean(np.max(y_proba, axis=1) if len(y_proba.shape) == 2 else y_proba))

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

        misclassified_points = X_2d[misclassified_mask]
        centroid = np.mean(misclassified_points, axis=0)

        distances = [distance.euclidean(p, centroid) for p in misclassified_points]

        clusters = {
            'has_blind_spot': True,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'cluster_radius': np.mean(distances),
            'points_in_cluster': len(misclassified_points)
        }

        return clusters