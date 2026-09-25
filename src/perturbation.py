# src/perturbation.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

class PerturbationSimulator:
    """Real-time feature perturbation simulator for ML-IDS"""
    
    def __init__(self, model, scaler, feature_names, class_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.class_names = class_names
    
    def get_sample_data(self, X_test, y_test, index):
        """Get a specific sample from test data"""
        sample = X_test[index]
        true_label = y_test[index]
        return sample, true_label
    
    def predict_with_perturbation(self, sample):
        """Get prediction for perturbed sample"""
        # Reshape for prediction
        sample_2d = sample.reshape(1, -1)
        
        # Predict
        pred = self.model.predict(sample_2d)[0]
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(sample_2d)[0]
        else:
            proba = None
        
        return pred, proba
    
    def create_perturbation_ui(self, sample, feature_names, original_pred, original_proba):
        """Create interactive sliders for each feature"""
        
        modified_sample = sample.copy()
        changes_made = []
        
        st.markdown("### 🔧 Adjust Features (Move the sliders)")
        st.markdown("Watch how the prediction changes in real-time!")
        
        # Create a grid of sliders (3 columns)
        cols = st.columns(3)
        
        for idx, feature in enumerate(feature_names[:30]):  # Limit to top 30 features
            col_idx = idx % 3
            with cols[col_idx]:
                # Get min/max from original data (approximate)
                min_val = float(np.percentile(sample, 5)) if idx == 0 else 0
                max_val = float(np.percentile(sample, 95)) if idx == 0 else 100
            
                new_val = st.slider(
                    f"{feature[:20]}",
                    min_value=0.0,
                    max_value=float(sample[idx] * 5),
                    value=float(sample[idx]),
                    step=float(sample[idx] / 100) if sample[idx] != 0 else 0.01,
                    key=f"slider_{idx}"
                )
                
                if new_val != sample[idx]:
                    changes_made.append(feature)
                
                modified_sample[idx] = new_val
        
        return modified_sample, changes_made
    
    def display_prediction(self, pred, proba, original_pred, original_proba, class_names):
        """Show prediction results with comparison"""
        
        pred_name = class_names[pred] if pred < len(class_names) else str(pred)
        original_pred_name = class_names[original_pred] if original_pred < len(class_names) else str(original_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Current Prediction")
            if pred == original_pred:
                st.success(f"✅ **{pred_name}** (Same as original)")
            else:
                st.error(f"⚠️ **{pred_name}** (Changed from {original_pred_name})")
            
            if proba is not None:
                st.progress(float(max(proba)))
                st.caption(f"Confidence: {max(proba):.2%}")
        
        with col2:
            st.markdown("### 📋 Original Prediction")
            st.info(f"**{original_pred_name}**")
            if original_proba is not None:
                st.caption(f"Confidence: {max(original_proba):.2%}")
    
    def find_vulnerable_features(self, sample, feature_names):
        """Find which features most affect the prediction"""
        
        vulnerabilities = []
        original_pred, original_proba = self.predict_with_perturbation(sample)
        
        # Test each feature
        for idx, feature in enumerate(feature_names[:20]):  # Test top 20
            modified = sample.copy()
            
            # Increase by 50%
            modified[idx] = modified[idx] * 1.5
            new_pred, _ = self.predict_with_perturbation(modified)
            
            if new_pred != original_pred:
                vulnerabilities.append({
                    'feature': feature,
                    'change': '+50%',
                    'effect': 'Prediction flipped!'
                })
            else:
                # Try decreasing by 50%
                modified[idx] = sample[idx] * 0.5
                new_pred, _ = self.predict_with_perturbation(modified)
                
                if new_pred != original_pred:
                    vulnerabilities.append({
                        'feature': feature,
                        'change': '-50%',
                        'effect': 'Prediction flipped!'
                    })
        
        return vulnerabilities