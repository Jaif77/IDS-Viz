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
                confidence = np.max(y_proba, axis=1)
            else:
                confidence = y_proba
        else:
            confidence = np.ones(len(y_true))
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame()
        plot_df['x'] = X_2d[:, 0].tolist()
        plot_df['y'] = X_2d[:, 1].tolist()
        plot_df['True Label'] = y_true
        plot_df['Predicted'] = y_pred
        plot_df['Correct'] = (y_true == y_pred)
        plot_df['Confidence'] = confidence.tolist() if hasattr(confidence, 'tolist') else confidence
        
        # ========== Handle class names with NaN safety ==========
        if class_names is not None:
            if hasattr(class_names, 'tolist'):
                class_names = class_names.tolist()
            elif not isinstance(class_names, list):
                class_names = list(class_names)
            class_names = [str(c) for c in class_names]
        else:
            class_names = None
        
        def get_class_name(x):
            try:
                if pd.isna(x):
                    return 'Unknown'
                idx = int(float(x))
                if class_names is not None and 0 <= idx < len(class_names):
                    return str(class_names[idx])
                return str(idx)
            except:
                return str(x)
        
        plot_df['True Label Name'] = plot_df['True Label'].apply(get_class_name)
        plot_df['Predicted Name'] = plot_df['Predicted'].apply(get_class_name)
        
        # Ensure all labels are strings
        plot_df['True Label Name'] = plot_df['True Label Name'].astype(str)
        plot_df['Predicted Name'] = plot_df['Predicted Name'].astype(str)
        
        fig = go.Figure()
        
        # ========== CORRECT PREDICTIONS ==========
        correct_df = plot_df[plot_df['Correct'] == True].copy()
        
        if len(correct_df) > 0:
            correct_df['Label_Upper'] = correct_df['True Label Name'].str.upper()
            benign_correct = correct_df[correct_df['Label_Upper'].str.contains('BENIGN', na=False)]
            attack_correct = correct_df[~correct_df['Label_Upper'].str.contains('BENIGN', na=False)]
        else:
            benign_correct = pd.DataFrame()
            attack_correct = pd.DataFrame()
        
        # Benign points
        if len(benign_correct) > 0:
            fig.add_trace(go.Scatter(
                x=benign_correct['x'].tolist(),
                y=benign_correct['y'].tolist(),
                mode='markers',
                name='Benign (Correct)',
                marker=dict(
                    size=8,
                    color='#2ECC71',
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate='<b>✅ CORRECT (Benign)</b><br>' +
                              '<b>True:</b> %{customdata[0]}<br>' +
                              '<b>Predicted:</b> %{customdata[1]}<extra></extra>',
                customdata=benign_correct[['True Label Name', 'Predicted Name']].values.tolist()
            ))
        
        # Attack points
        if len(attack_correct) > 0:
            fig.add_trace(go.Scatter(
                x=attack_correct['x'].tolist(),
                y=attack_correct['y'].tolist(),
                mode='markers',
                name='Attack (Correct)',
                marker=dict(
                    size=8,
                    color='#E74C3C',
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate='<b>✅ CORRECT (Attack)</b><br>' +
                              '<b>True:</b> %{customdata[0]}<br>' +
                              '<b>Predicted:</b> %{customdata[1]}<extra></extra>',
                customdata=attack_correct[['True Label Name', 'Predicted Name']].values.tolist()
            ))
        
        # ========== MISCLASSIFICATIONS ==========
        wrong_df = plot_df[plot_df['Correct'] == False].copy()
        
        if len(wrong_df) > 0:
            fig.add_trace(go.Scatter(
                x=wrong_df['x'].tolist(),
                y=wrong_df['y'].tolist(),
                mode='markers',
                name='⚠️ Misclassified',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='#F1C40F',
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>⚠️ MISCLASSIFIED</b><br>' +
                              '<b>True:</b> %{customdata[0]}<br>' +
                              '<b>Predicted:</b> %{customdata[1]}<extra></extra>',
                customdata=wrong_df[['True Label Name', 'Predicted Name']].values.tolist()
            ))
        
        # ========== Layout ==========
        fig.update_layout(
            title=f'Decision Boundary Visualization ({method_name})',
            xaxis=dict(
                title='Component 1',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.5)'
            ),
            yaxis=dict(
                title='Component 2',
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(128, 128, 128, 0.5)'
            ),
            hovermode='closest',
            legend=dict(
                x=0.01, y=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            height=650,
            plot_bgcolor='rgba(240, 240, 240, 0.1)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
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