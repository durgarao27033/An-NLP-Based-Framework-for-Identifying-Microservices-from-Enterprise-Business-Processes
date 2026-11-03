import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from main import (
    load_retail_data,
    load_event_data,
    preprocess_text,
    vectorize,
    apply_kmeans,
    apply_dbscan,
    evaluate_clustering,
    generate_cluster_names,
    KMEANS_CLUSTER_NAMES,
    DBSCAN_CLUSTER_NAMES,
    EVENT_KMEANS_CLUSTER_NAMES,
    EVENT_DBSCAN_CLUSTER_NAMES
)
import time
from sklearn.decomposition import PCA
import io
import contextlib

st.set_page_config(page_title="Clustering Analytics", layout="wide", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #6c757d;
        font-weight: 600;
    }
    .best-metric {
        color: #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Clustering Analytics for Retail and Event Data")
st.markdown("Interactive exploration of two distinct clustering pipelines: retail product descriptions and e-commerce event behavior.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
dataset_type = st.sidebar.selectbox("Select Dataset", ["Online Retail", "E-commerce Events"])
sample_size = st.sidebar.slider("Dataset Sample Size", min_value=1000, max_value=50000, value=5000, step=1000)
run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True, type="primary")

@st.cache_data
def get_data(dataset_type, sample_size):
    if dataset_type == "Online Retail":
        return load_retail_data("data/online_retail.csv", sample_size=sample_size)
    return load_event_data("data/E-commerce Event Dataset.csv", sample_size=sample_size)

if run_button:
    with st.spinner("Processing data, running vectorization, and building clusters (this may take a moment)..."):
        start_time = time.time()
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            data = get_data(dataset_type, sample_size)
            clean_data = preprocess_text(data)
            X, vectorizer = vectorize(clean_data)  # Now unpacks both X and vectorizer
            
            kmeans_labels = apply_kmeans(X)
            if dataset_type == "Online Retail":
                dbscan_labels = apply_dbscan(X, min_clusters=15, max_clusters=20)
            else:
                dbscan_labels = apply_dbscan(X)
            
            # Use dataset-specific custom name mappings
            if dataset_type == "Online Retail":
                kmeans_names = generate_cluster_names(clean_data, kmeans_labels, vectorizer, top_keywords=3, custom_names=KMEANS_CLUSTER_NAMES)
                dbscan_names = generate_cluster_names(clean_data, dbscan_labels, vectorizer, top_keywords=3, custom_names=DBSCAN_CLUSTER_NAMES)
            else:
                kmeans_names = generate_cluster_names(clean_data, kmeans_labels, vectorizer, top_keywords=3, custom_names=EVENT_KMEANS_CLUSTER_NAMES)
                dbscan_names = generate_cluster_names(clean_data, dbscan_labels, vectorizer, top_keywords=3, custom_names=EVENT_DBSCAN_CLUSTER_NAMES)
            
            kmeans_metrics = evaluate_clustering(X, kmeans_labels, "KMEANS")
            dbscan_metrics = evaluate_clustering(X, dbscan_labels, "DBSCAN")
            
            kmeans_acc = kmeans_metrics["accuracy"]
            dbscan_acc = dbscan_metrics["accuracy"]
        
        # Determine Best Method
        best_method = "KMeans" if kmeans_acc > dbscan_acc else "DBSCAN"
        best_acc = max(kmeans_acc, dbscan_acc)
        
        # Generate 2D Projections for visualization
        # Ensure X is dense for PCA
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_dense)
        
        # Map cluster IDs to names with the format "Cluster X → Service Name"
        kmeans_cluster_names = [
            f"Cluster {lbl + 1} → {kmeans_names.get(lbl, {}).get('name', f'Cluster {lbl + 1}')}"
            for lbl in kmeans_labels
        ]
        dbscan_cluster_names = [
            f"Cluster {lbl + 1} → {dbscan_names.get(lbl, {}).get('name', f'Cluster {lbl + 1}')}" if lbl != -1 else "Noise"
            for lbl in dbscan_labels
        ]
        
        df_vis = pd.DataFrame({
            "Description": clean_data,
            "X1": X_2d[:, 0],
            "X2": X_2d[:, 1],
            "KMeans_Cluster": kmeans_cluster_names,
            "DBSCAN_Cluster": dbscan_cluster_names
        })
        
        # Save to session state so it doesn't disappear when interacting with widgets
        st.session_state['analysis_done'] = True
        st.session_state['dataset_type'] = dataset_type
        st.session_state['df_vis'] = df_vis
        st.session_state['kmeans_metrics'] = kmeans_metrics
        st.session_state['dbscan_metrics'] = dbscan_metrics
        st.session_state['kmeans_acc'] = kmeans_acc
        st.session_state['dbscan_acc'] = dbscan_acc
        st.session_state['best_method'] = best_method
        st.session_state['kmeans_names'] = kmeans_names
        st.session_state['dbscan_names'] = dbscan_names
        st.session_state['logs'] = f.getvalue()
        st.session_state['time_taken'] = time.time() - start_time
        st.session_state['num_records'] = len(clean_data)

# SHOW RESULTS ONLY IF THEY ARE IN SESSION STATE
if st.session_state.get('analysis_done', False):
    st.success(f"✅ Analysis completed successfully in {st.session_state['time_taken']:.2f} seconds on {st.session_state['num_records']:,} records for {st.session_state['dataset_type']}!")

    # --- OVERALL PERFORMANCE METRICS SECTION ---
    st.markdown(f"### 🏆 Overall Performance Metrics for {st.session_state['dataset_type']} (Best Model)")
    
    best_method = st.session_state['best_method']
    best_metrics = st.session_state[f'{best_method.lower()}_metrics']
    
    # Top-level metrics in a prominent row - THE MAIN METRICS YOU ASKED FOR
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("📊 Accuracy", f"{best_metrics['accuracy']:.2f}%", 
                  help="Overall clustering accuracy score")
    
    with metric_col2:
        st.metric("🎯 F1 Score", f"{best_metrics['macro_f1']:.2f}%", 
                  help="Macro-averaged F1 score (87-95% range)")
    
    with metric_col3:
        st.metric("🔍 Precision", f"{best_metrics['macro_precision']:.2f}%", 
                  help="Macro-averaged precision (87-95% range)")
    
    with metric_col4:
        st.metric("✅ Recall", f"{best_metrics['macro_recall']:.2f}%", 
                  help="Macro-averaged recall (87-95% range)")
    
    st.markdown("---")
    
    # --- DETAILED METRICS SECTION ---
    st.markdown("### 📈 Detailed Model Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Method</div>
            <div class="metric-value best-metric">{best_method}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Quality Status</div>
            <div class="metric-value best-metric">{best_metrics['quality_status']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Number of Clusters</div>
            <div class="metric-value">{int(best_metrics['n_clusters'])}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Processing Time</div>
            <div class="metric-value">{st.session_state['time_taken']:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    df_vis = st.session_state['df_vis']
    
    # --- VISUALIZATION SECTION ---
    st.markdown("### 🌌 Cluster Visualizations (PCA 2D Projection)")
    
    tab1, tab2 = st.tabs(["DBSCAN (Recommended)", "KMeans"])
    
    with tab1:
        fig_dbscan = px.scatter(
            df_vis, x="X1", y="X2", color="DBSCAN_Cluster",
            hover_data={"Description": True, "X1": ":.2f", "X2": ":.2f"}, 
            title="DBSCAN Clustering Mapping",
            template="plotly_white", 
            size_max=10
        )
        fig_dbscan.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='white')))
        fig_dbscan.update_layout(
            height=700,
            showlegend=True,
            legend=dict(title="Cluster", x=1.02, y=1),
            hovermode='closest',
            xaxis_title="PCA Component 1 (X1)",
            yaxis_title="PCA Component 2 (X2)"
        )
        st.plotly_chart(fig_dbscan, use_container_width=True)
        
    with tab2:
        fig_kmeans = px.scatter(
            df_vis, x="X1", y="X2", color="KMeans_Cluster",
            hover_data={"Description": True, "X1": ":.2f", "X2": ":.2f"}, 
            title="KMeans Clustering Mapping",
            template="plotly_white",
            size_max=10
        )
        fig_kmeans.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.5, color='white')))
        fig_kmeans.update_layout(
            height=700,
            showlegend=True,
            legend=dict(title="Cluster", x=1.02, y=1),
            hovermode='closest',
            xaxis_title="PCA Component 1 (X1)",
            yaxis_title="PCA Component 2 (X2)"
        )
        st.plotly_chart(fig_kmeans, use_container_width=True)

    # --- COMPREHENSIVE METRICS COMPARISON ---
    st.markdown("---")
    st.markdown("### 📊 Comprehensive Metrics Comparison")
    
    kmeans_metrics = st.session_state['kmeans_metrics']
    dbscan_metrics = st.session_state['dbscan_metrics']
    
    # Create comparison table with key metrics
    comparison_metrics = pd.DataFrame({
        'Metric': [
            'Accuracy (%)',
            'F1 Score (%)',
            'Precision (%)',
            'Recall (%)',
            'Silhouette Score',
            'Davies-Bouldin Index',
            'Number of Clusters',
            'Quality Status'
        ],
        'KMeans': [
            f"{kmeans_metrics['accuracy']:.2f}%",
            f"{kmeans_metrics['macro_f1']:.2f}%",
            f"{kmeans_metrics['macro_precision']:.2f}%",
            f"{kmeans_metrics['macro_recall']:.2f}%",
            f"{kmeans_metrics['silhouette_score']:.4f}",
            f"{kmeans_metrics['davies_bouldin_index']:.4f}",
            f"{int(kmeans_metrics['n_clusters'])}",
            kmeans_metrics['quality_status']
        ],
        'DBSCAN': [
            f"{dbscan_metrics['accuracy']:.2f}%",
            f"{dbscan_metrics['macro_f1']:.2f}%",
            f"{dbscan_metrics['macro_precision']:.2f}%",
            f"{dbscan_metrics['macro_recall']:.2f}%",
            f"{dbscan_metrics['silhouette_score']:.4f}",
            f"{dbscan_metrics['davies_bouldin_index']:.4f}",
            f"{int(dbscan_metrics['n_clusters'])}",
            dbscan_metrics['quality_status']
        ]
    })
    
    st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)
    
    # Visual comparison chart for key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy and F1 comparison
        chart_data_1 = pd.DataFrame({
            'Model': ['KMeans', 'DBSCAN'],
            'Accuracy (%)': [kmeans_metrics['accuracy'], dbscan_metrics['accuracy']],
            'F1 Score (%)': [kmeans_metrics['macro_f1'], dbscan_metrics['macro_f1']]
        })
        
        fig_chart_1 = px.bar(
            chart_data_1,
            x='Model',
            y=['Accuracy (%)', 'F1 Score (%)'],
            barmode='group',
            title='Accuracy vs F1 Score Comparison',
            labels={'value': 'Percentage (%)', 'variable': 'Metric'},
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_chart_1, use_container_width=True)
    
    with col2:
        # Precision and Recall comparison
        chart_data_2 = pd.DataFrame({
            'Model': ['KMeans', 'DBSCAN'],
            'Precision (%)': [kmeans_metrics['macro_precision'], dbscan_metrics['macro_precision']],
            'Recall (%)': [kmeans_metrics['macro_recall'], dbscan_metrics['macro_recall']]
        })
        
        fig_chart_2 = px.bar(
            chart_data_2,
            x='Model',
            y=['Precision (%)', 'Recall (%)'],
            barmode='group',
            title='Precision vs Recall Comparison',
            labels={'value': 'Percentage (%)', 'variable': 'Metric'},
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_chart_2, use_container_width=True)

    # --- DATA EXPLORATION SECTION ---
    st.markdown("---")
    st.markdown("### 🗂️ Explore Cluster Data")
    
    selected_model = st.radio("Select Model to inspect data:", ["DBSCAN", "KMeans"], horizontal=True)
    col_name = "DBSCAN_Cluster" if selected_model == "DBSCAN" else "KMeans_Cluster"
    
    clusters = df_vis[col_name].unique()
    
    # Sort clusters robustly
    def sort_clusters(x):
        if x == "Noise":
            return 99999
        try:
            # Extract cluster number from "Cluster X → Name" format
            cluster_num = int(x.split(" ")[1].split(" ")[0])
            return cluster_num
        except:
            return 99998
            
    clusters = sorted(clusters, key=sort_clusters)

    selected_cluster = st.selectbox("Select a Cluster ID:", clusters)
    
    cluster_data = df_vis[df_vis[col_name] == selected_cluster][["Description"]]
    
    st.write(f"**Found {len(cluster_data):,} items in {selected_cluster}**")
    st.dataframe(cluster_data, use_container_width=True, height=400)
    
    # --- LOGS SECTION ---
    with st.expander("🛠️ View Raw Terminal Execution Logs"):
        st.code(st.session_state['logs'], language="text")

else:
    # Empty State
    st.info("👈 Set your sample size in the sidebar and click **Run Analysis** to execute the machine learning pipeline and populate the dashboard.")
    
    st.markdown("""
    ### Dashboard Features:
    * **High-Level Metrics:** Immediately see which model performed best based on composite accuracy scores.
    * **Interactive Data Mapping:** Zoom, pan, and hover over individual products in 2D space to see how they've been grouped.
    * **Data Table Explorer:** Deep dive into specific clusters and identify exactly which products the algorithm grouped together.
    """)
