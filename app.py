"""
Delhivery Graph Intelligence — Streamlit Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(
    page_title = "Delhivery Network Intelligence",
    page_icon  = "🚚",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    border-left: 4px solid #1D9E75;
}
.danger-card  { border-left: 4px solid #E24B4A; }
.warning-card { border-left: 4px solid #EF9F27; }
.section-header {
    font-size: 20px;
    font-weight: bold;
    padding: 10px 0;
    border-bottom: 2px solid #333;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {}
    try:
        data['hub_df']       = pd.read_csv('outputs/hub_metrics.csv')
        data['delay_df']     = pd.read_csv('outputs/delay_corridors.csv')
        data['edge_stats']   = pd.read_csv('outputs/edge_stats.csv')
        data['processed_df'] = pd.read_csv('outputs/processed_df.csv')
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}. Run all notebooks first.")
        st.stop()
    return data

@st.cache_resource
def load_graph():
    try:
        with open('outputs/graph.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("graph.pkl not found. Run notebook 01 first.")
        st.stop()

@st.cache_resource
def load_models():
    models = {}
    try:
        with open('outputs/graph_model.pkl', 'rb') as f:
            models['eta_model'] = pickle.load(f)
        with open('outputs/ftl_model.pkl', 'rb') as f:
            models['ftl_model'] = pickle.load(f)
        with open('outputs/ftl_features.pkl', 'rb') as f:
            models['ftl_features'] = pickle.load(f)
    except FileNotFoundError:
        pass
    return models

data   = load_data()
G      = load_graph()
models = load_models()

hub_df       = data['hub_df']
delay_df     = data['delay_df']
edge_stats   = data['edge_stats']
processed_df = data['processed_df']


# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/5f/Delhivery_Logo.png",
                 width=160, use_column_width=False)
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🔴 Bottleneck Hubs",
    "🛣️ Delay Corridors",
    "📈 Model Benchmark",
    "🚛 FTL vs Carting",
    "🗺️ Network Map",
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Nodes (Hubs):** {G.number_of_nodes():,}")
st.sidebar.markdown(f"**Edges (Corridors):** {G.number_of_edges():,}")
pct_delayed = len(delay_df) / G.number_of_edges() * 100
st.sidebar.markdown(f"**Delay Corridors:** {len(delay_df):,} ({pct_delayed:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🚚 Delhivery — Network Operations Intelligence")
    st.markdown("Graph-based analytics for smarter ETA predictions and network bottleneck identification.")

    st.markdown("---")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hubs", f"{G.number_of_nodes():,}")
    with col2:
        st.metric("Total Corridors", f"{G.number_of_edges():,}")
    with col3:
        st.metric("Chronic Delay Corridors", f"{len(delay_df):,}",
                  delta=f"{pct_delayed:.1f}% of all routes", delta_color="inverse")
    with col4:
        if 'is_delayed' in processed_df.columns:
            breach = processed_df['is_delayed'].mean() * 100
        else:
            breach = processed_df['segment_delay_ratio'].gt(1.2).mean() * 100
        st.metric("Overall SLA Breach Rate", f"{breach:.1f}%", delta_color="inverse")

    st.markdown("---")

    # Two-column layout
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔴 Top 5 Bottleneck Hubs")
        top5 = hub_df.head(5)[['hub_name', 'betweenness',
                                'avg_breach_rate', 'risk_score']].copy()
        top5['avg_breach_rate'] = top5['avg_breach_rate'].map('{:.1%}'.format)
        top5['betweenness']     = top5['betweenness'].map('{:.4f}'.format)
        top5['risk_score']      = top5['risk_score'].map('{:.4f}'.format)
        top5.index = range(1, 6)
        st.dataframe(top5, use_container_width=True)

    with col_right:
        st.markdown("#### 🛣️ Top 5 Worst Delay Corridors")
        top5_corr = delay_df.head(5)[['source', 'destination',
                                       'delay_ratio', 'breach_rate']].copy()
        top5_corr['delay_ratio']  = top5_corr['delay_ratio'].map('{:.2f}x'.format)
        top5_corr['breach_rate']  = top5_corr['breach_rate'].map('{:.1%}'.format)
        top5_corr.index = range(1, 6)
        st.dataframe(top5_corr, use_container_width=True)

    # Risk score distribution
    st.markdown("---")
    st.markdown("#### Risk Score Distribution Across Hubs")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(hub_df['risk_score'], bins=50, color='#E24B4A', alpha=0.8, edgecolor='white')
    ax.axvline(hub_df['risk_score'].quantile(0.9), color='orange',
               linestyle='--', linewidth=2, label='Top 10% threshold')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Number of Hubs')
    ax.set_title('Hub Risk Score Distribution')
    ax.legend()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE 2: BOTTLENECK HUBS
# ═══════════════════════════════════════════════════════════════
elif page == "🔴 Bottleneck Hubs":
    st.title("🔴 Bottleneck Hub Analysis")
    st.markdown("Hubs ranked by risk score — betweenness centrality + breach rate + degree.")

    n_hubs = st.slider("Show top N hubs", min_value=5, max_value=50, value=20)

    top_hubs = hub_df.head(n_hubs).copy()

    # Color code by risk
    def risk_color(score):
        if score > 0.3:  return "🔴 Critical"
        if score > 0.15: return "🟠 High"
        if score > 0.05: return "🟡 Moderate"
        return "🟢 Low"

    top_hubs['risk_level'] = top_hubs['risk_score'].apply(risk_color)
    top_hubs['avg_breach_rate'] = top_hubs['avg_breach_rate'].map('{:.1%}'.format)

    st.dataframe(
        top_hubs[['hub_name', 'risk_level', 'betweenness',
                  'in_degree', 'out_degree', 'clustering', 'avg_breach_rate', 'risk_score']],
        use_container_width=True,
        height=500
    )

    # Scatter: betweenness vs breach rate
    st.markdown("---")
    st.markdown("#### Betweenness vs Breach Rate — Each dot is a hub")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        hub_df['betweenness'],
        hub_df['avg_breach_rate'],
        c=hub_df['risk_score'],
        cmap='RdYlGn_r',
        s=hub_df['in_degree'] * 5 + 10,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.3
    )
    plt.colorbar(scatter, ax=ax, label='Risk Score')
    for _, row in hub_df.head(8).iterrows():
        ax.annotate(str(row['hub_name'])[:20],
                    (row['betweenness'], row['avg_breach_rate']),
                    fontsize=7, alpha=0.9)
    ax.set_xlabel('Betweenness Centrality')
    ax.set_ylabel('Avg Breach Rate')
    ax.set_title('Hub Risk Map — Size = In-Degree, Color = Risk Score')
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE 3: DELAY CORRIDORS
# ═══════════════════════════════════════════════════════════════
elif page == "🛣️ Delay Corridors":
    st.title("🛣️ Chronic Delay Corridor Audit")

    col1, col2 = st.columns(2)
    with col1:
        min_delay = st.slider("Min delay ratio filter", 1.0, 3.0, 1.2, 0.1)
    with col2:
        if 'route_type' in delay_df.columns:
            rt_filter = st.multiselect("Route type",
                                        delay_df['route_type'].unique().tolist(),
                                        default=delay_df['route_type'].unique().tolist())
        else:
            rt_filter = None

    filtered = delay_df[delay_df['delay_ratio'] >= min_delay].copy()
    if rt_filter and 'route_type' in filtered.columns:
        filtered = filtered[filtered['route_type'].isin(rt_filter)]

    st.markdown(f"**{len(filtered)} corridors** match your filters")

    display_cols = ['source', 'destination', 'delay_ratio',
                    'breach_rate', 'trip_count']
    if 'route_type' in filtered.columns:
        display_cols.append('route_type')
    if 'median_distance' in filtered.columns:
        display_cols.append('median_distance')

    st.dataframe(
        filtered[display_cols].head(100).style.background_gradient(
            subset=['delay_ratio', 'breach_rate'], cmap='Reds'
        ),
        use_container_width=True,
        height=450
    )

    # Bar chart: worst 15 corridors
    st.markdown("---")
    st.markdown("#### Top 15 Worst Corridors by Delay Ratio")
    top15 = delay_df.head(15).copy()
    top15['label'] = top15['source'].str[:12] + ' → ' + top15['destination'].str[:12]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#E24B4A' if x > 1.5 else '#EF9F27' for x in top15['delay_ratio']]
    ax.barh(top15['label'], top15['delay_ratio'], color=colors, edgecolor='white')
    ax.axvline(1.2, color='green', linestyle='--', linewidth=2, label='20% threshold')
    ax.set_xlabel('Median Delay Ratio (actual / OSRM)')
    ax.set_title('Top 15 Chronic Delay Corridors', fontweight='bold')
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE 4: MODEL BENCHMARK
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Model Benchmark":
    st.title("📈 ETA Model Benchmark — The Graph Advantage")

    if os.path.exists('outputs/model_benchmark.csv'):
        results = pd.read_csv('outputs/model_benchmark.csv')
        st.dataframe(results.style.highlight_min(subset=['MAE', 'RMSE'], color='#d4edda')
                                   .highlight_max(subset=['R²', '% Within 15%'], color='#d4edda'),
                     use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#4a4a6a', '#EF9F27', '#1D9E75']
            ax.bar(results['Model'], results['MAE'], color=colors, edgecolor='white')
            ax.set_title('MAE Comparison (lower = better)', fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(results['Model'], results['% Within 15%'], color=colors, edgecolor='white')
            ax.set_title('% Trips Within 15% of Actual (higher = better)', fontweight='bold')
            ax.tick_params(axis='x', rotation=15)
            st.pyplot(fig)
            plt.close()

        if os.path.exists('outputs/feature_importance.png'):
            st.markdown("#### Feature Importance — Graph Model")
            st.image('outputs/feature_importance.png')
    else:
        st.warning("Run notebook 02_eta_model.ipynb first to generate benchmark results.")


# ═══════════════════════════════════════════════════════════════
# PAGE 5: FTL VS CARTING
# ═══════════════════════════════════════════════════════════════
elif page == "🚛 FTL vs Carting":
    st.title("🚛 FTL vs Carting Decision Framework")

    if os.path.exists('outputs/ftl_tradeoff.csv'):
        tradeoff = pd.read_csv('outputs/ftl_tradeoff.csv')

        st.markdown("#### Cost-Time Tradeoff by Distance Bucket")
        if os.path.exists('outputs/ftl_cost_time_tradeoff.png'):
            st.image('outputs/ftl_cost_time_tradeoff.png')

        st.markdown("---")
        st.markdown("#### Model Evaluation")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('outputs/ftl_model_eval.png'):
                st.image('outputs/ftl_model_eval.png')
        with col2:
            if os.path.exists('outputs/shap_summary.png'):
                st.image('outputs/shap_summary.png')
            elif os.path.exists('outputs/ftl_feature_importance.png'):
                st.image('outputs/ftl_feature_importance.png')

        # Live predictor
        st.markdown("---")
        st.markdown("#### 🔮 Try the Route Recommender")
        if models.get('ftl_model') and models.get('ftl_features'):
            col1, col2, col3 = st.columns(3)
            with col1:
                distance = st.number_input("Distance (km)", 10.0, 3000.0, 300.0)
            with col2:
                hour = st.slider("Hour of day", 0, 23, 10)
            with col3:
                delay_ratio_inp = st.number_input("Historical delay ratio", 0.5, 3.0, 1.1)

            if st.button("Recommend Route Type"):
                input_data = pd.DataFrame([{
                    f: 0 for f in models['ftl_features']
                }])
                if 'segment_osrm_distance' in input_data.columns:
                    input_data['segment_osrm_distance'] = distance
                if 'hour' in input_data.columns:
                    input_data['hour'] = hour
                if 'delay_ratio' in input_data.columns:
                    input_data['delay_ratio'] = delay_ratio_inp

                prob = models['ftl_model'].predict_proba(input_data)[0][1]
                recommendation = "FTL" if prob > 0.5 else "Carting"
                confidence = max(prob, 1 - prob) * 100

                if recommendation == "FTL":
                    st.success(f"✅ Recommended: **FTL** (confidence: {confidence:.1f}%)")
                    st.info("FTL is better for this corridor — faster and lower SLA breach risk.")
                else:
                    st.info(f"💡 Recommended: **Carting** (confidence: {confidence:.1f}%)")
                    st.info("Carting is cost-effective for this corridor with acceptable delay risk.")
        else:
            st.warning("Run notebook 03_ftl_carting.ipynb first to enable the recommender.")
    else:
        st.warning("Run notebook 03_ftl_carting.ipynb first.")


# ═══════════════════════════════════════════════════════════════
# PAGE 6: NETWORK MAP
# ═══════════════════════════════════════════════════════════════
elif page == "🗺️ Network Map":
    st.title("🗺️ Live Network Map")

    if os.path.exists('outputs/bottleneck_network.html'):
        st.markdown("**Interactive network — zoom, drag, hover for details**")
        st.markdown("""
        - 🔴 **Red nodes** = Critical bottleneck hubs
        - 🟠 **Amber nodes** = Moderate risk
        - 🟢 **Green nodes** = Low risk
        - **Red edges** = Severely delayed corridors (>1.5x OSRM)
        - **Amber edges** = Delayed corridors (1.2–1.5x OSRM)
        """)
        with open('outputs/bottleneck_network.html', 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.warning("Run notebook 01_graph_bottleneck.ipynb first to generate the network map.")
        st.info("The pyvis interactive map is saved as outputs/bottleneck_network.html")

        # Static fallback
        st.markdown("#### Static Network Preview")
        if os.path.exists('outputs/network_static.png'):
            st.image('outputs/network_static.png')
