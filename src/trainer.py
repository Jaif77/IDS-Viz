# src/trainer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class IDSTrainer:
    """Core ML training engine for IDS-Viz"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_multiclass = False
        self.class_names = None
        
    def prepare_data(self, df, target_column=' Label'):
        """Prepare dataframe for training (handles both binary and multi-class)"""
        
        data = df.copy()
        
        # ========== CRITICAL FIX: Handle column names with spaces ==========
        for col in data.columns:
            if col != col.strip():
                new_col = col.strip()
                data.rename(columns={col: new_col}, inplace=True)
                print(f"✅ Renamed column '{col}' to '{new_col}'")
        
        # ========== FIX: Use correct label column ==========
        # Try these possible label column names in order
        possible_label_columns = ['Label', 'Attack_Type', 'attack_type', 'label', 'Class']
        
        label_col = None
        for col in possible_label_columns:
            if col in data.columns:
                label_col = col
                print(f"✅ Found label column: '{label_col}'")
                break
        
        # If still not found, use the last column
        if label_col is None:
            label_col = data.columns[-1]
            print(f"⚠️ Using last column as label: '{label_col}'")
        
        # Separate features and target
        X = data.drop(columns=[label_col])
        y = data[label_col]
        
        # ========== DEBUG: Print label info ==========
        print(f"🔍 DEBUG: y dtype: {y.dtype}")
        print(f"🔍 DEBUG: y unique values: {y.unique()}")
        print(f"🔍 DEBUG: y value counts:\n{y.value_counts()}")
        
        self.feature_names = X.columns.tolist()
        
        # Handle infinite and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        self.feature_names = numeric_cols.tolist()
        
        # ========== Multi-class support ==========
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
            self.is_multiclass = len(self.class_names) > 2
            y = y_encoded
        else:
            unique_vals = np.unique(y)
            if len(unique_vals) > 2:
                self.class_names = [str(v) for v in unique_vals]
                self.is_multiclass = True
            else:
                y = (y != 0).astype(int)
                self.class_names = ['Benign', 'Attack']
                self.is_multiclass = False
        
        print(f"🔍 DEBUG: Number of unique classes: {len(np.unique(y))}")
        print(f"🔍 DEBUG: Class names: {self.class_names}")
        print(f"🔍 DEBUG: Is multiclass: {self.is_multiclass}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='Random Forest'):
        """Train selected model"""
        if model_type == 'Random Forest':
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'Decision Tree':
            self.model = DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        elif model_type == 'Logistic Regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )# ← multi_class removed
    
        self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Get all evaluation metrics"""
        y_pred = self.model.predict(X_test)
        
        # ========== DEBUG: Print prediction info ==========
        print(f"🔍 DEBUG: Unique in y_test: {np.unique(y_test)}")
        print(f"🔍 DEBUG: Unique in y_pred: {np.unique(y_pred)}")
        print(f"🔍 DEBUG: y_test shape: {y_test.shape}")
        print(f"🔍 DEBUG: y_pred shape: {y_pred.shape}")
        
        if self.is_multiclass:
            # Only use classification_report if we have both classes
            if len(np.unique(y_test)) > 1:
                report = classification_report(
                    y_test, y_pred, 
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0
                )
            else:
                report = {}
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': report,
                'is_multiclass': True
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
                'is_multiclass': False
            }
        
        cm = confusion_matrix(y_test, y_pred)
        return metrics, y_pred, cm
    
    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        return None
    
    def get_class_names(self):
        """Return class names for display"""
        if self.class_names is not None:
            return self.class_names
        return ['Benign', 'Attack']