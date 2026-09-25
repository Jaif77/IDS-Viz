import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import time
import sys
import os

# Add src folder to path
sys.path.append(os.path.dirname(__file__))

from src.trainer import IDSTrainer
from src.decision_boundary import DecisionBoundaryExplorer
from src.perturbation import PerturbationSimulator

# Page configuration
st.set_page_config(
    page_title="IDS-Viz", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .css-1aumxhk {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = IDSTrainer()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'y_test_boundary' not in st.session_state:
    st.session_state.y_test_boundary = None
if 'y_pred_boundary' not in st.session_state:
    st.session_state.y_pred_boundary = None

# ==================== HELPER FUNCTION ====================
def find_label_column(df):
    """Find the label column in a dataframe"""
    possible_names = ['Label', ' Label', 'label', ' label', 
                     'Attack_Type', ' Attack_Type', 'attack_type',
                     'Class', ' Class', 'class', ' class']
    
    # Try exact matches first
    for name in possible_names:
        if name in df.columns:
            return name
    
    # Try keyword search
    for col in df.columns:
        if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower():
            return col
    
    # Fallback to last column
    return df.columns[-1]

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.title("🛡️ IDS-Viz")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📂 Data Upload", "⚙️ Model Training", "🔬 Decision Boundary", "🔧 Perturbation Lab", "📄 Reports"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**About**")
    st.info(
        "Visualize and analyze vulnerabilities in "
        "ML-based Intrusion Detection Systems"
    )

# ==================== PAGE 1: DATA UPLOAD ====================
if page == "📂 Data Upload":
    st.title("📂 1. Load Network Traffic Data")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Source")
        
        st.markdown("**📁 Local Datasets**")
        
        cic_path = "data/MachineLearningCVE/"
        available_files = []
        
        if os.path.exists(cic_path):
            available_files = [f for f in os.listdir(cic_path) if f.endswith('.csv')]
        
        if available_files:
            selected_file = st.selectbox(
                "Choose CIC-IDS2017 file:",
                available_files
            )
            
            if st.button("📥 Load Selected File", use_container_width=True):
                with st.spinner("Loading dataset..."):
                    file_path = os.path.join(cic_path, selected_file)
                    df = pd.read_csv(file_path, nrows=200000)
                    
                    # Strip column names
                    df.columns = df.columns.str.strip()
                    
                    # Find label column
                    label_col = find_label_column(df)
                    
                    st.session_state.df = df
                    st.session_state.selected_file = selected_file
                    st.success(f"✅ Loaded {len(df):,} rows from {selected_file}")
                    st.info(f"📋 Label column: `{label_col}`")
        else:
            st.warning("No CIC-IDS2017 files found in data/MachineLearningCVE/")
        
        st.markdown("---")
        st.markdown("**📤 Upload CSV**")
        uploaded_file = st.file_uploader(
            "Or upload your own file:",
            type=['csv'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Strip column names
            df.columns = df.columns.str.strip()
            
            # Find label column
            label_col = find_label_column(df)
            
            st.session_state.df = df
            st.session_state.selected_file = uploaded_file.name
            st.success(f"✅ Loaded {len(df):,} rows from {uploaded_file.name}")
            st.info(f"📋 Label column: `{label_col}`")
    
    with col2:
        if st.session_state.df is not None:
            df = st.session_state.df
            
            st.subheader(f"📊 Dataset: {st.session_state.selected_file}")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Samples", f"{len(df):,}")
            with metric_col2:
                st.metric("Features", len(df.columns))
            
            # ========== FIX: Use helper function for label detection ==========
            label_col = find_label_column(df)
            
            st.caption(f"📋 Using label column: `{label_col}`")
            
            unique_classes = df[label_col].unique()
            is_multiclass = len(unique_classes) > 2
            
            if is_multiclass:
                st.metric("Number of Classes", len(unique_classes))
                st.caption(f"Classes: {', '.join([str(c)[:15] for c in unique_classes[:5]])}")
                if len(unique_classes) > 5:
                    st.caption(f"... and {len(unique_classes)-5} more")
                
                # Multi-class pie chart
                st.subheader("Class Distribution")
                
                class_counts = df[label_col].value_counts()
                
                fig = px.pie(
                    values=class_counts.values,
                    names=class_counts.index.astype(str),
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show bar chart
                fig_bar = px.bar(
                    x=class_counts.index.astype(str),
                    y=class_counts.values,
                    title="Class Distribution (Bar Chart)",
                    color=class_counts.values,
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(height=400, xaxis_title="Class", yaxis_title="Count")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # Binary classification
                if df[label_col].dtype == 'object':
                    attack_count = sum(df[label_col] != 'BENIGN')
                else:
                    attack_count = sum(df[label_col] != 0)
                
                benign_count = len(df) - attack_count
                
                with metric_col3:
                    st.metric("Attack Samples", f"{attack_count:,}")
                    st.caption(f"{attack_count/len(df)*100:.1f}%")
                with metric_col4:
                    st.metric("Benign Samples", f"{benign_count:,}")
                    st.caption(f"{benign_count/len(df)*100:.1f}%")
                
                st.subheader("Class Distribution")
                
                fig = px.pie(
                    values=[benign_count, attack_count],
                    names=['Benign', 'Attack'],
                    color_discrete_sequence=['#2ECC71', '#E74C3C'],
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🔍 Preview Data (First 100 rows)"):
                st.dataframe(df.head(100), use_container_width=True)
                
            with st.expander("📈 Column Statistics"):
                st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("👈 Load a dataset to begin analysis")
            st.markdown("---")
            st.markdown("""
            ### 🎯 What is IDS-Viz?
            
            **IDS-Viz** helps you understand why ML-based Intrusion Detection Systems fail.
            
            **You can:**
            1. Load network traffic data (CIC-IDS2017 or your own)
            2. Train ML models with one click
            3. Visualize decision boundaries
            4. Simulate feature perturbations
            5. Identify vulnerabilities
            
            **Start by loading a dataset from the left panel.**
            """)

# ==================== PAGE 2: MODEL TRAINING ====================
elif page == "⚙️ Model Training":
    st.title("⚙️ 2. Train & Evaluate IDS Model")
    st.markdown("---")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first (Go to Data Upload page)")
        st.stop()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Select Model Architecture:",
            ["Random Forest", "Decision Tree", "Logistic Regression"],
            help="Random Forest: Best accuracy, slower\nDecision Tree: Fast, interpretable\nLogistic Regression: Simple baseline"
        )
        
        st.markdown("**Training Parameters**")
        sample_size = st.slider(
            "Training samples (rows):",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=1000,
            help="Use more rows to include attacks."
        )
        
        st.markdown("---")
        train_button = st.button(
            "🚀 START TRAINING",
            use_container_width=True,
            type="primary"
        )
        
        if train_button:
            with st.spinner("🔄 Training model..."):
                try:
                    # ========== FIX: Use random sample instead of head ==========
                    if len(st.session_state.df) > sample_size:
                        df = st.session_state.df.sample(n=sample_size, random_state=42)
                    else:
                        df = st.session_state.df
            
                    trainer = st.session_state.trainer
                    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
                    model = trainer.train_model(X_train, y_train, model_type)
                    metrics, y_pred, cm = trainer.evaluate_model(X_test, y_test)
            
                    st.session_state.metrics = metrics
                    st.session_state.cm = cm
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.model_trained = True
                    st.session_state.current_model = model_type
            
                    st.success("✅ Model trained successfully!")
            
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    with col2:
        if st.session_state.model_trained:
            st.subheader(f"📊 Model Performance: {st.session_state.current_model}")
            
            if st.session_state.metrics.get('is_multiclass', False):
                st.subheader("📊 Multi-Class Classification Report")
                
                report = st.session_state.metrics['classification_report']
                report_df = pd.DataFrame(report).T.round(3)
                st.dataframe(report_df, use_container_width=True)
                
                st.metric("Overall Accuracy", f"{st.session_state.metrics['accuracy']:.3f}")
                
                st.subheader("🎯 Per-Class Performance")
                class_names = st.session_state.trainer.get_class_names()
                
                cols = st.columns(min(4, len(class_names)))
                for idx, class_name in enumerate(class_names):
                    if class_name in report:
                        col_idx = idx % len(cols)
                        with cols[col_idx]:
                            st.markdown(f"**{class_name}**")
                            st.metric("Precision", f"{report[class_name]['precision']:.3f}")
                            st.metric("Recall", f"{report[class_name]['recall']:.3f}")
                            st.metric("F1", f"{report[class_name]['f1-score']:.3f}")
            else:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Accuracy", f"{st.session_state.metrics['accuracy']:.3f}")
                with m2:
                    st.metric("Precision", f"{st.session_state.metrics['precision']:.3f}")
                with m3:
                    st.metric("Recall", f"{st.session_state.metrics['recall']:.3f}")
                with m4:
                    st.metric("F1-Score", f"{st.session_state.metrics['f1_score']:.3f}")
            
            st.subheader("Confusion Matrix")
            
            cm = st.session_state.cm
            
            if st.session_state.metrics.get('is_multiclass', False):
                class_names = st.session_state.trainer.get_class_names()
                
                if hasattr(class_names, 'tolist'):
                    class_names = class_names.tolist()
                elif not isinstance(class_names, list):
                    class_names = list(class_names)
                
                cm_display = cm.tolist()
                
                n_classes = len(cm_display)
                if len(class_names) != n_classes:
                    class_names = [f"Class {i}" for i in range(n_classes)]
                
                fig = px.imshow(
                    cm_display,
                    x=class_names,
                    y=class_names,
                    color_continuous_scale='Blues',
                    text_auto=True,
                    aspect="auto"
                )
                fig.update_layout(height=500, title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 Rows = Actual, Columns = Predicted. Diagonal = correct.")
                
            elif cm.shape == (2, 2):
                fig = px.imshow(
                    cm,
                    x=['Predicted Benign', 'Predicted Attack'],
                    y=['Actual Benign', 'Actual Attack'],
                    color_continuous_scale='Blues',
                    text_auto=True,
                    aspect="auto"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                tn, fp, fn, tp = cm.ravel()
                
                st.subheader("📋 Analysis")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**✅ Strengths**")
                    st.markdown(f"• Correctly detected **{tp}** attacks")
                    st.markdown(f"• Correctly allowed **{tn}** benign flows")
                    
                with col_b:
                    st.markdown("**⚠️ Weaknesses**")
                    st.markdown(f"• Missed **{fn}** attacks (False Negatives)")
                    st.markdown(f"• False alarms on **{fp}** benign flows (False Positives)")
            else:
                st.warning("⚠️ Test data contains only one class. Cannot show full confusion matrix.")
                st.info("💡 Try increasing the training sample size to include attacks.")
                
                fig = px.imshow(
                    [[cm[0,0]]],
                    x=['Only Class'],
                    y=['Only Class'],
                    color_continuous_scale='Blues',
                    text_auto=True,
                    aspect="auto"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                if cm.size == 1:
                    st.metric("Total Samples", cm[0, 0])
                    st.caption("All samples belong to the same class. Try loading more rows.")
            
            if st.session_state.current_model in ['Random Forest', 'Decision Tree']:
                st.subheader("🎯 Feature Importance")
                importance = st.session_state.trainer.get_feature_importance()
                
                if importance:
                    top_features = dict(sorted(importance.items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:15])
                    
                    fig = px.bar(
                        x=list(top_features.values()),
                        y=list(top_features.keys()),
                        orientation='h',
                        title="Top 15 Most Important Features",
                        color=list(top_features.values()),
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Configure and train your first model")

# ==================== PAGE 3: DECISION BOUNDARY ====================
elif page == "🔬 Decision Boundary":
    st.title("🔬 Decision Boundary Explorer")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first (Go to Model Training page)")
        st.stop()
    
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns([2, 2, 2, 1])
    
    with col_filter1:
        show_benign = st.checkbox("Show Benign (Correct)", value=True)
    with col_filter2:
        show_attack = st.checkbox("Show Attack (Correct)", value=True)
    with col_filter3:
        show_misclassified = st.checkbox("Show Misclassified", value=True)
    with col_filter4:
        export_btn = st.button("📥 Export Misclassified", use_container_width=True)
    
    if export_btn:
        try:
            export_limit = st.slider("Rows to export:", 1000, 50000, 10000)
            df_sample = st.session_state.df.head(export_limit)
            trainer = st.session_state.trainer
            
            X_train, X_test, y_train, y_test = trainer.prepare_data(df_sample)
            y_pred = trainer.model.predict(X_test)
            
            explorer = DecisionBoundaryExplorer(
                trainer.model, trainer.scaler, trainer.feature_names
            )
            
            misclassified_df = explorer.get_misclassified_samples(X_test, y_test, y_pred, trainer.feature_names)
            
            csv = misclassified_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="misclassified_samples.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            st.info("Try using a smaller sample size or load fewer rows from your dataset.")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        
        method = st.radio(
            "Reduction Method:",
            ["PCA (Faster)", "t-SNE (Better but Slower)"],
            help="PCA is quick. t-SNE shows clearer clusters but takes longer."
        )
        method_clean = "pca" if "PCA" in method else "tsne"
        
        viz_samples = st.slider(
            "Samples to visualize:",
            min_value=500,
            max_value=10000,
            value=3000,
            step=500,
            help="More samples = better picture but slower to render"
        )
        
        visualize_btn = st.button("🎨 Generate Decision Map", use_container_width=True, type="primary")
    
    with col2:
        if visualize_btn:
            with st.spinner("Generating decision boundary map... This may take 30-60 seconds"):
                try:
                    # ========== FIX 1: Random sample use korun ==========
                    if len(st.session_state.df) > viz_samples:
                        df = st.session_state.df.sample(n=viz_samples, random_state=42)
                    else:
                        df = st.session_state.df
            
                    trainer = st.session_state.trainer
            
                    # Prepare data (scaler notun kore fit hobe)
                    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
            
                    # ========== FIX 2: MODEL RETRAIN KORUN ==========
                    model_type = st.session_state.get('current_model', 'Random Forest')
                    trainer.train_model(X_train, y_train, model_type)
            
                    # Ekhon model notun data diye trained
                    explorer = DecisionBoundaryExplorer(
                        trainer.model, 
                        trainer.scaler, 
                        trainer.feature_names
                    )
            
                    X_2d, method_name = explorer.reduce_dimensions(
                        X_test, 
                        method=method_clean
                    )
            
                    y_pred = trainer.model.predict(X_test)
                    
                    if hasattr(trainer.model, 'predict_proba'):
                        y_proba = trainer.model.predict_proba(X_test)
                        if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                            y_proba = y_proba[:, 1]
                    else:
                        y_proba = None
                    
                    class_names = trainer.get_class_names()
                    
                    fig = explorer.create_scatter_plot(
                        X_2d, y_test, y_pred, 
                        y_proba=y_proba,
                        class_names=class_names,
                        method_name=method_name.upper()
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    stats = explorer.get_statistics(X_2d, y_test, y_pred, y_proba)
                    
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                    with col_a:
                        st.metric("Total Samples", stats['total_samples'])
                    with col_b:
                        st.metric("✅ Correct", stats['correct_count'])
                    with col_c:
                        st.metric("❌ Misclassified", stats['wrong_count'])
                    with col_d:
                        st.metric("📊 Accuracy", f"{stats['accuracy']:.1%}")
                    if 'avg_confidence' in stats:
                        with col_e:
                            st.metric("🎯 Avg Confidence", f"{stats['avg_confidence']:.1%}")
                    
                    clusters = explorer.get_cluster_regions(X_2d, y_test, y_pred)
                    
                    if clusters and clusters['has_blind_spot'] and stats['wrong_count'] > 5:
                        st.warning(f"⚠️ **Blind Spot Detected!** {clusters['points_in_cluster']} misclassified points form a cluster.")
                    elif stats['wrong_count'] > 0:
                        st.info(f"💡 **Insight:** {stats['wrong_count']} samples were misclassified.")
                    else:
                        st.success("🎉 Perfect classification on this sample!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Try reducing the sample size or switching to PCA if t-SNE is too slow.")
        else:
            st.info("👈 Click **Generate Decision Map** to visualize where your model succeeds and fails")

# ==================== PAGE 4: PERTURBATION LAB ====================
elif page == "🔧 Perturbation Lab":
    st.title("🔧 Feature Perturbation Simulator")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first (Go to Model Training page)")
        st.stop()
    
    st.markdown("""
    **What does this do?**
    - Select any network flow from your test data
    - Move the sliders to change packet features
    - Watch the model's prediction change in **real-time**
    - Discover which features are most vulnerable to manipulation
    """)
    
    st.markdown("---")
    
    try:
        df_sample = st.session_state.df.head(10000)
        trainer = st.session_state.trainer
        
        X_train, X_test, y_train, y_test = trainer.prepare_data(df_sample)
        
        # ========== FIX: Convert to numpy arrays ==========
        X_test_array = np.array(X_test)
        y_test_array = np.array(y_test)
        
        y_pred = trainer.model.predict(X_test_array)
        y_pred_array = np.array(y_pred)
        
        class_names = trainer.get_class_names()
        
        sample_df = pd.DataFrame(X_test_array[:, :5], columns=trainer.feature_names[:5])
        sample_df['True Label'] = y_test_array
        sample_df['Predicted'] = y_pred_array
        sample_df['Correct'] = (y_test_array == y_pred_array)
        
        st.subheader("1. Select a Network Flow")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_filter = st.radio(
                "Show:",
                ["All Samples", "Only Misclassified", "Only Correct"],
                index=0
            )
        
        with col2:
            if sample_filter == "Only Misclassified":
                filtered_df = sample_df[sample_df['Correct'] == False]
            elif sample_filter == "Only Correct":
                filtered_df = sample_df[sample_df['Correct'] == True]
            else:
                filtered_df = sample_df
            
            if len(filtered_df) == 0:
                st.warning(f"No samples found for filter: {sample_filter}")
                st.info("Try selecting 'All Samples' instead.")
                st.stop()
            
            sample_index = st.selectbox(
                f"Select a sample ({len(filtered_df)} available):",
                range(len(filtered_df)),
                format_func=lambda x: f"Sample {x} | True: {filtered_df.iloc[x]['True Label']} | Pred: {filtered_df.iloc[x]['Predicted']}"
            )
        
        if sample_filter == "All Samples":
            actual_index = sample_index
        else:
            actual_index = filtered_df.index[sample_index]
        
        st.markdown("---")
        
        sample = X_test_array[actual_index]
        true_label = y_test_array[actual_index]
        original_pred = y_pred_array[actual_index]
        
        if hasattr(trainer.model, 'predict_proba'):
            original_proba = trainer.model.predict_proba(sample.reshape(1, -1))[0]
        else:
            original_proba = None
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("True Label", class_names[true_label] if true_label < len(class_names) else str(true_label))
        with col_b:
            st.metric("Original Prediction", class_names[original_pred] if original_pred < len(class_names) else str(original_pred))
        with col_c:
            if original_proba is not None:
                st.metric("Confidence", f"{max(original_proba):.2%}")
        
        st.markdown("---")
        
        st.subheader("2. Modify Features (Move the Sliders)")
        
        importance = trainer.get_feature_importance()
        
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
            top_feature_names = [f[0] for f in top_features]
            
            st.info(f"💡 Showing top {len(top_feature_names)} most important features.")
            
            modified_sample = sample.copy()
            cols = st.columns(2)
            changes_made = []
            
            for idx, feature_name in enumerate(top_feature_names):
                try:
                    feature_idx = trainer.feature_names.index(feature_name)
                except ValueError:
                    continue
                
                col_idx = idx % 2
                with cols[col_idx]:
                    original_val = sample[feature_idx]
                    min_val = max(0, original_val * 0.1)
                    max_val = original_val * 10 if original_val > 0 else 100
                    
                    new_val = st.slider(
                        f"📊 {feature_name[:30]}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(original_val),
                        step=float(original_val / 50) if original_val != 0 else 0.01,
                        key=f"perturb_{feature_idx}"
                    )
                    
                    if new_val != original_val:
                        changes_made.append(feature_name)
                    
                    modified_sample[feature_idx] = new_val
            
            st.markdown("---")
            
            new_pred = trainer.model.predict(modified_sample.reshape(1, -1))
            new_pred_class = new_pred[0]
            
            new_proba = None
            if hasattr(trainer.model, 'predict_proba'):
                new_proba = trainer.model.predict_proba(modified_sample.reshape(1, -1))[0]
            
            st.subheader("3. Result")
            
            if new_pred_class == original_pred:
                st.success(f"✅ Prediction remains: **{class_names[new_pred_class]}**")
            else:
                st.error(f"⚠️ PREDICTION FLIPPED! From **{class_names[original_pred]}** to **{class_names[new_pred_class]}**")
                
                if changes_made:
                    st.warning(f"📌 Changes made to: {', '.join(changes_made[:5])}")
            
            if new_proba is not None:
                st.progress(float(max(new_proba)))
                st.caption(f"New confidence: {max(new_proba):.2%}")
            
            with st.expander("📊 See all feature changes"):
                comparison_data = []
                for feature_name in top_feature_names[:10]:
                    try:
                        feature_idx = trainer.feature_names.index(feature_name)
                        comparison_data.append({
                            'Feature': feature_name,
                            'Original': f"{sample[feature_idx]:.4f}",
                            'Modified': f"{modified_sample[feature_idx]:.4f}",
                            'Change': f"{(modified_sample[feature_idx] - sample[feature_idx]) / (sample[feature_idx] + 0.001) * 100:.1f}%"
                        })
                    except:
                        pass
                
                if comparison_data:
                    st.table(pd.DataFrame(comparison_data))
        else:
            st.info("Train a tree-based model first to see feature importance.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try using a smaller sample size or retrain your model.")

# ==================== PAGE 5: REPORTS ====================
else:
    st.title("📄 Generate Analysis Report")
    st.markdown("---")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first (Go to Model Training page)")
        st.stop()
    
    st.markdown("""
    **Generate a comprehensive PDF report** containing:
    - Dataset information
    - Model configuration
    - Performance metrics (Accuracy, Precision, Recall, F1)
    - Confusion matrix
    - Feature importance analysis
    - Key findings and recommendations
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Report Settings")
        
        report_title = st.text_input("Report Title", value="IDS-Viz Vulnerability Analysis Report")
        
        include_metrics = st.checkbox("Include Performance Metrics", value=True)
        include_cm = st.checkbox("Include Confusion Matrix", value=True)
        include_features = st.checkbox("Include Feature Importance", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        st.markdown("---")
        
        generate_btn = st.button("📄 Generate PDF Report", use_container_width=True, type="primary")
    
    with col2:
        if generate_btn:
            with st.spinner("Generating report... Please wait"):
                try:
                    from src.report_generator import ReportGenerator
                    
                    rg = ReportGenerator()
                    
                    dataset_info = {
                        'name': st.session_state.selected_file if st.session_state.selected_file else "Custom Dataset",
                        'total_samples': len(st.session_state.df),
                        'features': len(st.session_state.df.columns),
                        'num_classes': len(st.session_state.trainer.get_class_names())
                    }
                    
                    feature_importance = st.session_state.trainer.get_feature_importance()
                    
                    report_buffer = rg.create_report(
                        metrics=st.session_state.metrics,
                        cm=st.session_state.cm,
                        feature_importance=feature_importance,
                        dataset_info=dataset_info,
                        model_type=st.session_state.current_model,
                        is_multiclass=st.session_state.metrics.get('is_multiclass', False)
                    )
                    
                    st.success("✅ Report generated successfully!")
                    st.markdown("---")
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=report_buffer,
                        file_name=f"ids_viz_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.info("💡 **Tip:** Save this report for your documentation and defense presentation.")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
                    st.info("Make sure you have installed reportlab: `pip install reportlab`")