"""
Delhivery Network Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title = "Delhivery | Network Intelligence",
    page_icon  = "assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Delhivery Brand CSS ────────────────────────────────────────
st.markdown("""
<style>
/* ── Root variables ── */
:root {
    --dlv-red:    #E8192C;
    --dlv-dark:   #111111;
    --dlv-gray:   #1E1E1E;
    --dlv-mid:    #2A2A2A;
    --dlv-border: #333333;
    --dlv-text:   #F0F0F0;
    --dlv-muted:  #888888;
    --dlv-white:  #FFFFFF;
}

/* ── Global ── */
html, body, [class*="css"] {
    background-color: var(--dlv-dark);
    color: var(--dlv-text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0rem !important; }

/* ── Top header bar ── */
.dlv-header {
    background: var(--dlv-dark);
    border-bottom: 3px solid var(--dlv-red);
    padding: 16px 0 12px 0;
    margin-bottom: 28px;
}
.dlv-logo {
    font-size: 28px;
    font-weight: 900;
    letter-spacing: -1px;
    color: var(--dlv-white);
}
.dlv-logo span { color: var(--dlv-red); }
.dlv-subtitle {
    font-size: 12px;
    color: var(--dlv-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Section title ── */
.section-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--dlv-red);
    border-bottom: 1px solid var(--dlv-border);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; }
.kpi-card {
    flex: 1;
    background: var(--dlv-gray);
    border: 1px solid var(--dlv-border);
    border-top: 3px solid var(--dlv-red);
    border-radius: 4px;
    padding: 20px 24px;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--dlv-muted);
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: var(--dlv-white);
    line-height: 1;
}
.kpi-sub {
    font-size: 12px;
    color: var(--dlv-muted);
    margin-top: 6px;
}
.kpi-alert { border-top-color: var(--dlv-red); }
.kpi-alert .kpi-value { color: var(--dlv-red); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--dlv-gray) !important;
    border-right: 1px solid var(--dlv-border);
}
[data-testid="stSidebar"] * { color: var(--dlv-text) !important; }
.sidebar-logo {
    font-size: 22px;
    font-weight: 900;
    letter-spacing: -0.5px;
    padding: 8px 0 4px 0;
    border-bottom: 2px solid var(--dlv-red);
    margin-bottom: 20px;
}
.sidebar-logo span { color: var(--dlv-red); }
.sidebar-meta {
    font-size: 11px;
    color: var(--dlv-muted);
    margin-bottom: 4px;
    letter-spacing: 0.5px;
}

/* ── Tables ── */
[data-testid="stDataFrame"] {
    background: var(--dlv-gray) !important;
    border: 1px solid var(--dlv-border) !important;
    border-radius: 4px !important;
}

/* ── Radio buttons in sidebar ── */
[data-testid="stRadio"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* ── Divider ── */
.dlv-divider {
    border: none;
    border-top: 1px solid var(--dlv-border);
    margin: 20px 0;
}

/* ── Alert banner ── */
.dlv-alert {
    background: rgba(232,25,44,0.1);
    border: 1px solid rgba(232,25,44,0.4);
    border-left: 4px solid var(--dlv-red);
    border-radius: 3px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 13px;
    color: var(--dlv-text);
}

/* ── Info box ── */
.dlv-info {
    background: var(--dlv-mid);
    border: 1px solid var(--dlv-border);
    border-radius: 4px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.6;
}

/* ── Legend row ── */
.legend-row { display: flex; gap: 20px; margin-bottom: 14px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--dlv-muted); }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }

/* ── Button ── */
.stButton > button {
    background: var(--dlv-red) !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 28px !important;
}
.stButton > button:hover {
    background: #c41525 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    out = {}
    required = {
        'hub_df':       'outputs/hub_metrics.csv',
        'delay_df':     'outputs/delay_corridors.csv',
        'edge_stats':   'outputs/edge_stats.csv',
        'processed_df': 'outputs/processed_df.csv',
    }
    missing = [k for k,v in required.items() if not os.path.exists(v)]
    if missing:
        st.error(f"Missing output files: {missing}. Run all notebooks first.")
        st.stop()
    for k,v in required.items():
        out[k] = pd.read_csv(v)
    return out

@st.cache_resource
def load_graph():
    path = 'outputs/graph.pkl'
    if not os.path.exists(path):
        st.error("graph.pkl not found. Run notebook 01 first.")
        st.stop()
    with open(path,'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_models():
    m = {}
    for name, path in [
        ('eta_model',    'outputs/graph_model.pkl'),
        ('ftl_model',    'outputs/ftl_model.pkl'),
        ('ftl_features', 'outputs/ftl_features.pkl'),
    ]:
        if os.path.exists(path):
            with open(path,'rb') as f:
                m[name] = pickle.load(f)
    return m

data   = load_data()
G      = load_graph()
models = load_models()

hub_df       = data['hub_df']
delay_df     = data['delay_df']
edge_stats   = data['edge_stats']
processed_df = data['processed_df']

# Pre-compute summary stats
total_nodes     = G.number_of_nodes()
total_edges     = G.number_of_edges()
delay_count     = len(delay_df)
delay_pct       = delay_count / total_edges * 100
if 'is_delayed' in processed_df.columns:
    breach_rate = processed_df['is_delayed'].mean() * 100
elif 'segment_delay_ratio' in processed_df.columns:
    breach_rate = (processed_df['segment_delay_ratio'] > 1.2).mean() * 100
else:
    breach_rate = 0.0


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        DELHI<span>VERY</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-meta">NETWORK INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-meta">Operations Analytics — Internal</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        options=[
            "Network Overview",
            "Bottleneck Analysis",
            "Delay Corridors",
            "Model Performance",
            "Route Type Framework",
            "Interactive Network",
        ],
        label_visibility="visible"
    )

    st.markdown("<hr style='border-color:#333;margin:20px 0'>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-meta">NETWORK STATS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-meta">Facilities &nbsp;&nbsp;&nbsp; {total_nodes:,}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-meta">Corridors &nbsp;&nbsp;&nbsp;&nbsp; {total_edges:,}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-meta">Delay Routes &nbsp; {delay_count:,} ({delay_pct:.1f}%)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-meta">Breach Rate &nbsp;&nbsp; {breach_rate:.1f}%</div>', unsafe_allow_html=True)


# ── Matplotlib theme ───────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1E1E1E',
    'axes.facecolor':   '#1E1E1E',
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#F0F0F0',
    'xtick.color':      '#888888',
    'ytick.color':      '#888888',
    'text.color':       '#F0F0F0',
    'grid.color':       '#2A2A2A',
    'grid.linewidth':   0.5,
})

DLV_RED   = '#E8192C'
DLV_GRAY  = '#444444'
DLV_LIGHT = '#888888'


# ═══════════════════════════════════════════════════════════════
# HEADER (all pages)
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="dlv-header">
    <div class="dlv-logo">DELHI<span>VERY</span></div>
    <div class="dlv-subtitle">Network Operations Intelligence</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: NETWORK OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "Network Overview":

    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Total Facilities</div>
            <div class="kpi-value">{total_nodes:,}</div>
            <div class="kpi-sub">Active nodes in network graph</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Active Corridors</div>
            <div class="kpi-value">{total_edges:,}</div>
            <div class="kpi-sub">Directed edges in logistics graph</div>
        </div>
        <div class="kpi-card kpi-alert">
            <div class="kpi-label">Chronic Delay Routes</div>
            <div class="kpi-value">{delay_count:,}</div>
            <div class="kpi-sub">{delay_pct:.1f}% of all corridors exceeding +20% OSRM</div>
        </div>
        <div class="kpi-card kpi-alert">
            <div class="kpi-label">SLA Breach Rate</div>
            <div class="kpi-value">{breach_rate:.1f}%</div>
            <div class="kpi-sub">Trips exceeding OSRM estimate by 20%+</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Top 5 Bottleneck Hubs</div>', unsafe_allow_html=True)
        top5 = hub_df.head(5)[['hub_name','betweenness','avg_breach_rate','risk_score']].copy()
        top5.columns = ['Hub', 'Betweenness', 'Breach Rate', 'Risk Score']
        top5['Breach Rate'] = top5['Breach Rate'].map('{:.1%}'.format)
        top5['Betweenness'] = top5['Betweenness'].map('{:.4f}'.format)
        top5['Risk Score']  = top5['Risk Score'].map('{:.4f}'.format)
        top5.index = range(1, 6)
        st.dataframe(top5, use_container_width=True, height=220)

    with col_r:
        st.markdown('<div class="section-title">Top 5 Worst Corridors</div>', unsafe_allow_html=True)
        top5c = delay_df.head(5)[['source','destination','delay_ratio','breach_rate']].copy()
        top5c.columns = ['Source', 'Destination', 'Delay Ratio', 'Breach Rate']
        top5c['Delay Ratio'] = top5c['Delay Ratio'].map('{:.2f}x'.format)
        top5c['Breach Rate'] = top5c['Breach Rate'].map('{:.1%}'.format)
        top5c.index = range(1, 6)
        st.dataframe(top5c, use_container_width=True, height=220)

    st.markdown('<div class="section-title">Risk Score Distribution Across All Hubs</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(hub_df['risk_score'], bins=60, color=DLV_RED, alpha=0.85, edgecolor='#111')
    ax.axvline(hub_df['risk_score'].quantile(0.9), color='#FF8C00',
               linestyle='--', lw=1.5, label='Top 10% risk threshold')
    ax.set_xlabel('Risk Score', fontsize=11)
    ax.set_ylabel('Number of Hubs', fontsize=11)
    ax.set_title('Hub Risk Score Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE: BOTTLENECK ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "Bottleneck Analysis":

    st.markdown('<div class="section-title">Bottleneck Hub Rankings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        n_hubs = st.slider("Hubs to display", 5, 100, 25)

    top_hubs = hub_df.head(n_hubs).copy()

    def risk_label(s):
        if s > 0.3:  return "Critical"
        if s > 0.15: return "High"
        if s > 0.05: return "Moderate"
        return "Low"

    top_hubs['Risk Level']    = top_hubs['risk_score'].apply(risk_label)
    top_hubs['Breach Rate']   = top_hubs['avg_breach_rate'].map('{:.1%}'.format)
    top_hubs['Betweenness']   = top_hubs['betweenness'].map('{:.4f}'.format)
    top_hubs['Risk Score']    = top_hubs['risk_score'].map('{:.4f}'.format)
    top_hubs['Clustering']    = top_hubs['clustering'].map('{:.3f}'.format)

    st.dataframe(
        top_hubs[['hub_name','Risk Level','Betweenness','in_degree',
                  'out_degree','Clustering','Breach Rate','Risk Score']]
        .rename(columns={'hub_name':'Hub','in_degree':'In','out_degree':'Out'}),
        use_container_width=True,
        height=480
    )

    st.markdown('<div class="section-title">Betweenness vs Breach Rate</div>', unsafe_allow_html=True)
    st.markdown('<div class="dlv-info">Each point is a facility. Size = in-degree (volume received). Position = structural risk vs operational delay rate.</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(
        hub_df['betweenness'],
        hub_df['avg_breach_rate'],
        c=hub_df['risk_score'],
        cmap='RdYlGn_r',
        s=hub_df['in_degree'] * 6 + 8,
        alpha=0.75,
        edgecolors='#111',
        linewidth=0.3
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Risk Score', fontsize=10)
    for _, row in hub_df.head(8).iterrows():
        ax.annotate(
            str(row['hub_name'])[:18],
            (row['betweenness'], row['avg_breach_rate']),
            fontsize=7.5,
            color='#F0F0F0',
            alpha=0.9
        )
    ax.axhline(0.2, color=DLV_RED, linestyle='--', lw=1, alpha=0.5, label='20% breach line')
    ax.set_xlabel('Betweenness Centrality — structural chokepoint score', fontsize=10)
    ax.set_ylabel('Average SLA Breach Rate', fontsize=10)
    ax.set_title('Hub Risk Map', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE: DELAY CORRIDORS
# ═══════════════════════════════════════════════════════════════
elif page == "Delay Corridors":

    st.markdown('<div class="section-title">Chronic Delay Corridor Audit</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        min_delay = st.slider("Minimum delay ratio", 1.0, 3.0, 1.2, 0.05)
    with col2:
        if 'route_type' in delay_df.columns:
            rt_opts = delay_df['route_type'].dropna().unique().tolist()
            rt_sel  = st.multiselect("Route type", rt_opts, default=rt_opts)
        else:
            rt_sel = None

    filtered = delay_df[delay_df['delay_ratio'] >= min_delay].copy()
    if rt_sel and 'route_type' in filtered.columns:
        filtered = filtered[filtered['route_type'].isin(rt_sel)]

    st.markdown(f"""
    <div class="dlv-alert">
        {len(filtered)} corridors match your criteria — {len(filtered)/total_edges*100:.1f}% of the entire network
    </div>
    """, unsafe_allow_html=True)

    disp_cols = ['source','destination','delay_ratio','breach_rate','trip_count']
    if 'route_type'      in filtered.columns: disp_cols.append('route_type')
    if 'median_distance' in filtered.columns: disp_cols.append('median_distance')

    st.dataframe(
        filtered[disp_cols].head(150)
        .rename(columns={
            'source':'Source','destination':'Destination',
            'delay_ratio':'Delay Ratio','breach_rate':'Breach Rate',
            'trip_count':'Trips','route_type':'Route','median_distance':'Dist (km)'
        }),
        use_container_width=True,
        height=400
    )

    st.markdown('<div class="section-title">Top 15 Worst Corridors</div>', unsafe_allow_html=True)

    top15 = delay_df.head(15).copy()
    top15['label'] = top15['source'].str[:14] + '  →  ' + top15['destination'].str[:14]

    fig, ax = plt.subplots(figsize=(13, 6))
    colors  = [DLV_RED if x > 1.5 else '#EF9F27' for x in top15['delay_ratio']]
    bars    = ax.barh(top15['label'], top15['delay_ratio'], color=colors, edgecolor='#111', height=0.65)
    ax.axvline(1.2, color='#4CAF50', linestyle='--', lw=1.5, label='+20% OSRM threshold')
    ax.axvline(1.5, color=DLV_RED,   linestyle=':',  lw=1,   label='+50% OSRM threshold')
    for bar, val in zip(bars, top15['delay_ratio']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}x', va='center', fontsize=9, color='#F0F0F0')
    ax.set_xlabel('Median Delay Ratio  (actual time / OSRM time)', fontsize=10)
    ax.set_title('Top 15 Chronic Delay Corridors', fontsize=13, fontweight='bold', pad=12)
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(axis='x')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown('<div class="section-title">ETA Prediction — Benchmark Results</div>', unsafe_allow_html=True)

    if os.path.exists('outputs/model_benchmark.csv'):
        results = pd.read_csv('outputs/model_benchmark.csv')

        st.dataframe(
            results.rename(columns={'Within_15pct':'Within 15%','R2':'R²'}),
            use_container_width=True,
            height=160
        )

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            colors  = [DLV_LIGHT, '#EF9F27', DLV_RED]
            bars    = ax.bar(results['Model'], results['MAE'], color=colors,
                             edgecolor='#111', width=0.5)
            for bar, val in zip(bars, results['MAE']):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        str(val), ha='center', fontsize=10, fontweight='bold')
            ax.set_title('Mean Absolute Error — Lower is Better', fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel('MAE')
            ax.tick_params(axis='x', rotation=12, labelsize=9)
            ax.grid(axis='y')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            col_name = 'Within_15pct' if 'Within_15pct' in results.columns else 'Within 15%'
            fig, ax  = plt.subplots(figsize=(7, 5))
            bars     = ax.bar(results['Model'], results[col_name], color=colors,
                              edgecolor='#111', width=0.5)
            for bar, val in zip(bars, results[col_name]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{val}%', ha='center', fontsize=10, fontweight='bold')
            ax.set_title('Trips Predicted Within 15% of Actual — Higher is Better', fontsize=11, fontweight='bold', pad=10)
            ax.set_ylabel('% Trips')
            ax.tick_params(axis='x', rotation=12, labelsize=9)
            ax.grid(axis='y')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        if os.path.exists('outputs/feature_importance.png'):
            st.markdown('<div class="section-title">Feature Importance — Graph-Enhanced Model</div>', unsafe_allow_html=True)
            st.image('outputs/feature_importance.png', use_column_width=True)

    else:
        st.markdown("""
        <div class="dlv-alert">
            Benchmark results not found. Run notebook 02_eta_model.ipynb to generate model comparison.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: ROUTE TYPE FRAMEWORK
# ═══════════════════════════════════════════════════════════════
elif page == "Route Type Framework":

    st.markdown('<div class="section-title">FTL vs Carting — Decision Framework</div>', unsafe_allow_html=True)

    if os.path.exists('outputs/ftl_tradeoff.csv'):

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists('outputs/ftl_eval.png'):
                st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
                st.image('outputs/ftl_eval.png', use_column_width=True)
        with col2:
            img = 'outputs/shap_summary.png' if os.path.exists('outputs/shap_summary.png') else 'outputs/ftl_importance.png'
            if os.path.exists(img):
                st.markdown('<div class="section-title">Feature Drivers</div>', unsafe_allow_html=True)
                st.image(img, use_column_width=True)

        if os.path.exists('outputs/ftl_tradeoff.png'):
            st.markdown('<div class="section-title">Cost-Time Tradeoff by Distance</div>', unsafe_allow_html=True)
            st.image('outputs/ftl_tradeoff.png', use_column_width=True)

        st.markdown('<div class="section-title">Route Recommender</div>', unsafe_allow_html=True)

        if models.get('ftl_model') and models.get('ftl_features'):
            st.markdown('<div class="dlv-info">Enter corridor parameters to get a data-backed route type recommendation.</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1: distance     = st.number_input("Distance (km)",   10.0,  3000.0, 300.0, step=10.0)
            with col2: hour         = st.slider("Departure hour",         0,     23,     10)
            with col3: delay_hist   = st.number_input("Historical delay ratio", 0.5, 3.0, 1.1, step=0.05)
            with col4: cutoff_f     = st.number_input("Cutoff factor",   0,     10,     0)

            if st.button("Get Recommendation"):
                inp = pd.DataFrame([{f: 0 for f in models['ftl_features']}])
                for col_nm, val in [
                    ('segment_osrm_distance', distance),
                    ('hour', hour),
                    ('segment_delay_ratio', delay_hist),
                    ('cutoff_factor', cutoff_f),
                ]:
                    if col_nm in inp.columns:
                        inp[col_nm] = val

                prob           = models['ftl_model'].predict_proba(inp)[0][1]
                recommendation = "FTL" if prob > 0.5 else "Carting"
                confidence     = max(prob, 1-prob) * 100

                if recommendation == "FTL":
                    st.markdown(f"""
                    <div class="dlv-alert" style="border-left-color:#4CAF50;background:rgba(76,175,80,0.1);">
                        <strong>Recommendation: FTL</strong> &nbsp;|&nbsp; Confidence: {confidence:.1f}%<br>
                        <span style="color:#888;font-size:12px;">Full truck load is optimal for this corridor profile — lower SLA breach risk justifies cost premium.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="dlv-info">
                        <strong>Recommendation: Carting</strong> &nbsp;|&nbsp; Confidence: {confidence:.1f}%<br>
                        <span style="color:#888;font-size:12px;">Shared carting is cost-effective for this corridor — delay risk is within acceptable SLA bounds.</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dlv-alert">
                FTL model not found. Run notebook 03_ftl_carting.ipynb to enable the recommender.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="dlv-alert">
            No tradeoff data found. Run notebook 03_ftl_carting.ipynb first.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: INTERACTIVE NETWORK
# ═══════════════════════════════════════════════════════════════
elif page == "Interactive Network":

    st.markdown('<div class="section-title">Live Logistics Network</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="legend-row">
        <div class="legend-item"><div class="legend-dot" style="background:#E8192C"></div>Critical bottleneck hub</div>
        <div class="legend-item"><div class="legend-dot" style="background:#EF9F27"></div>High risk hub</div>
        <div class="legend-item"><div class="legend-dot" style="background:#1D9E75"></div>Low risk hub</div>
        <div class="legend-item"><div class="legend-dot" style="background:#888;border-radius:0"></div>Delayed corridor (&gt;1.5x OSRM)</div>
    </div>
    <div class="dlv-info" style="margin-bottom:16px;">
        Zoom and drag to explore. Hover any node or edge for operational details.
        Node size reflects network centrality. Edge thickness reflects delay severity.
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists('outputs/bottleneck_network.html'):
        with open('outputs/bottleneck_network.html','r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=720, scrolling=False)
    else:
        st.markdown("""
        <div class="dlv-alert">
            Interactive network not found. Run notebook 01_graph_bottleneck.ipynb to generate it.<br>
            Requires: <code>pip install pyvis</code>
        </div>
        """, unsafe_allow_html=True)

        if os.path.exists('outputs/network_static.png'):
            st.markdown('<div class="section-title">Static Network Preview</div>', unsafe_allow_html=True)
            st.image('outputs/network_static.png', use_column_width=True)