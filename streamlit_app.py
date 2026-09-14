"""
Interactive EDA Dashboard — Dark & Colorful Edition
Companion app to EDA_Colab.ipynb (Data Prep & EDA assignment)

Run locally:
    pip install streamlit pandas numpy plotly scipy openpyxl
    streamlit run streamlit_app.py

For best visuals, also copy the .streamlit/config.toml file (provided separately)
into a folder named ".streamlit" next to this script — it sets Streamlit's native
dark theme. The app still looks good without it (custom CSS below handles most of it).
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# =======================================================================
# PAGE CONFIG & THEME
# =======================================================================
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Vivid, high-contrast colorway used across every chart
COLORWAY = ["#00F5D4", "#F15BB5", "#9B5DE5", "#00BBF9", "#FEE440", "#FF6B6B", "#3A86FF", "#FB5607"]
PLOTLY_TEMPLATE = "plotly_dark"

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1b1330 0%, #0d0d16 45%, #060608 100%);
        color: #F5F5FF;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14101f 0%, #0a0a12 100%);
        border-right: 1px solid #2a2340;
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #00F5D4, #9B5DE5, #F15BB5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    p, li, label, span { color: #E4E1F5 !important; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1c1633, #241a3d);
        border: 1px solid #3a2f5c;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 4px 18px rgba(155, 93, 229, 0.20);
    }
    div[data-testid="stMetric"] label { color: #B8AEE0 !important; }
    button[data-baseweb="tab"] {
        background-color: #161225;
        border-radius: 10px 10px 0 0;
        color: #cfc7ff !important;
        padding: 8px 16px;
        margin-right: 4px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #8338EC, #00BBF9);
        color: white !important;
    }
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .hero {
        padding: 22px 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(155,93,229,0.18), rgba(0,245,212,0.10));
        border: 1px solid #3a2f5c;
        margin-bottom: 18px;
    }
    .insight-card {
        background: linear-gradient(135deg, #1c1633, #221a38);
        border-left: 4px solid #00F5D4;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #8338EC, #00BBF9);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def style_fig(fig, title=None, height=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E4E1F5", size=13),
        title=dict(text=title, font=dict(size=18, color="#F5F5FF")) if title else None,
        margin=dict(l=30, r=30, t=60 if title else 30, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    if height:
        fig.update_layout(height=height)
    return fig


# =======================================================================
# DATA LOADING
# =======================================================================
DEFAULT_FILE = "Assign.xlsx"


@st.cache_data
def load_data(file):
    name = file.name if hasattr(file, "name") else file
    if str(name).endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def infer_column_types(df):
    cat_cols, num_cols = [], []
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "category" or df[c].nunique() <= 30:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return cat_cols, num_cols


def outlier_flags(series, n_std=3):
    mean, sd = series.mean(), series.std()
    return (series < mean - n_std * sd) | (series > mean + n_std * sd)


st.sidebar.markdown("##  EDA Dashboard")
st.sidebar.markdown("---")
st.sidebar.subheader(" Data")

default_exists = os.path.exists(DEFAULT_FILE)
uploaded_file = st.sidebar.file_uploader(
    "Upload a different file (optional)" if default_exists else "Upload Assign.xlsx / .csv",
    type=["xlsx", "csv"],
)

if uploaded_file is not None:
    source_name = uploaded_file.name
    df_raw = load_data(uploaded_file)
elif default_exists:
    source_name = DEFAULT_FILE
    df_raw = load_data(DEFAULT_FILE)
    st.sidebar.success(f"Auto-loaded {DEFAULT_FILE}")
else:
    st.markdown('<div class="hero"><h1> Interactive EDA Dashboard</h1>'
                '<p>Place <code>Assign.xlsx</code> next to this script, or upload a file in the sidebar, to begin.</p></div>',
                unsafe_allow_html=True)
    st.stop()

df = df_raw.copy()
auto_cat, auto_num = infer_column_types(df)

st.sidebar.markdown("---")
st.sidebar.subheader(" Column Roles")
cat_cols = st.sidebar.multiselect("Categorical columns", options=df.columns.tolist(), default=auto_cat)
num_cols = st.sidebar.multiselect(
    "Numeric columns", options=[c for c in df.columns if c not in cat_cols],
    default=[c for c in auto_num if c not in cat_cols],
)
for c in cat_cols:
    df[c] = df[c].astype("category")

st.sidebar.markdown("---")
st.sidebar.caption("")

# =======================================================================
# HERO HEADER
# =======================================================================
st.markdown(
    f"""<div class="hero">
        <h1> Interactive Exploratory Data Analysis</h1>
        <p>Dataset: <b>{source_name}</b> &nbsp;|&nbsp; {df.shape[0]} rows × {df.shape[1]} columns
        &nbsp;|&nbsp; {len(cat_cols)} categorical, {len(num_cols)} numeric columns</p>
    </div>""",
    unsafe_allow_html=True,
)

tabs = st.tabs([" Overview", " Data Cleaning", " Univariate", " Bivariate",
                 " Multivariate", " Statistical Tests", " Insights"])
tab_overview, tab_clean, tab_uni, tab_bi, tab_multi, tab_stats, tab_insights = tabs

# =======================================================================
# OVERVIEW
# =======================================================================
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))
    total_outliers = sum(outlier_flags(df[c]).sum() for c in num_cols) if num_cols else 0
    c5.metric("Outlier Points (3σ)", int(total_outliers))

    st.markdown("#### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("#### Summary Statistics")
    st.dataframe(df[num_cols].describe().T.style.background_gradient(cmap="viridis", axis=0), use_container_width=True)

# =======================================================================
# DATA CLEANING
# =======================================================================
with tab_clean:
    st.markdown("### 1️⃣ Data Types")
    st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), use_container_width=True)

    st.markdown("### 2️⃣ Missing Data")
    missing = df.isna().sum()
    miss_df = pd.DataFrame({"missing_count": missing, "missing_pct": (missing / len(df) * 100).round(2)})
    st.dataframe(miss_df, use_container_width=True)
    if missing.sum() == 0:
        st.success(" No missing values — the dataset is already complete.")
    else:
        fig = px.bar(miss_df[miss_df.missing_count > 0], y="missing_count", color_discrete_sequence=[COLORWAY[1]])
        st.plotly_chart(style_fig(fig, "Missing Values by Column"), use_container_width=True)

    st.markdown("### 3️⃣ Duplicate Rows")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        st.success(" No duplicate rows found.")
    else:
        st.warning(f"️ {dup_count} duplicate row(s) found.")
        if st.button("Drop duplicate rows"):
            df = df.drop_duplicates().reset_index(drop=True)
            st.success(f"Dropped — new shape: {df.shape}")

    st.markdown("### 4️⃣ Inconsistent Text / Categories")
    if cat_cols:
        chosen_cat = st.selectbox("Inspect unique values for:", cat_cols)
        vc = df[chosen_cat].value_counts().reset_index()
        vc.columns = [chosen_cat, "count"]
        fig = px.bar(vc, x=chosen_cat, y="count", color=chosen_cat, color_discrete_sequence=COLORWAY)
        st.plotly_chart(style_fig(fig, f"Value Counts — {chosen_cat}"), use_container_width=True)

    st.markdown("### 5️⃣ Outliers (3-standard-deviation rule)")
    if num_cols:
        outlier_counts = {c: int(outlier_flags(df[c]).sum()) for c in num_cols}
        oc_df = pd.Series(outlier_counts, name="outlier_count").sort_values(ascending=False).reset_index()
        oc_df.columns = ["column", "outlier_count"]
        fig = px.bar(oc_df, x="column", y="outlier_count", color="outlier_count",
                     color_continuous_scale="Plasma")
        st.plotly_chart(style_fig(fig, "Outlier Count per Numeric Column"), use_container_width=True)

        outlier_col = st.selectbox("Boxplot for column:", num_cols, key="outlier_col")
        fig2 = px.box(df, y=outlier_col, points="all", color_discrete_sequence=[COLORWAY[2]])
        st.plotly_chart(style_fig(fig2, f"Boxplot — {outlier_col}", height=420), use_container_width=True)

# =======================================================================
# UNIVARIATE
# =======================================================================
with tab_uni:
    st.markdown("### Univariate Analysis — one variable at a time")
    col_type = st.radio("Variable type", ["Numeric", "Categorical"], horizontal=True, key="uni_type")

    if col_type == "Numeric" and num_cols:
        col = st.selectbox("Choose a numeric column", num_cols, key="uni_num")
        bins = st.slider("Number of bins", 5, 60, 25)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col, nbins=bins, marginal="box", color_discrete_sequence=[COLORWAY[0]])
            st.plotly_chart(style_fig(fig, f"Distribution of {col}"), use_container_width=True)
        with c2:
            fig2 = px.violin(df, y=col, box=True, points="all", color_discrete_sequence=[COLORWAY[3]])
            st.plotly_chart(style_fig(fig2, f"Violin Plot — {col}"), use_container_width=True)

        desc = df[col].describe()
        skew, kurt = df[col].skew(), df[col].kurt()
        cols = st.columns(4)
        cols[0].metric("Mean", f"{desc['mean']:.2f}")
        cols[1].metric("Std Dev", f"{desc['std']:.2f}")
        cols[2].metric("Skewness", f"{skew:.2f}")
        cols[3].metric("Kurtosis", f"{kurt:.2f}")

    elif col_type == "Categorical" and cat_cols:
        col = st.selectbox("Choose a categorical column", cat_cols, key="uni_cat")
        c1, c2 = st.columns(2)
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        with c1:
            fig = px.bar(vc, x=col, y="count", color=col, color_discrete_sequence=COLORWAY)
            st.plotly_chart(style_fig(fig, f"Count Plot — {col}"), use_container_width=True)
        with c2:
            fig2 = px.pie(vc, names=col, values="count", hole=0.55, color_discrete_sequence=COLORWAY)
            st.plotly_chart(style_fig(fig2, f"Share of Records — {col}"), use_container_width=True)
    else:
        st.info("No columns of this type are selected in the sidebar.")

# =======================================================================
# BIVARIATE
# =======================================================================
with tab_bi:
    st.markdown("### Bivariate Analysis — relationship between two variables")
    combo = st.radio("Combination", ["Numeric vs Numeric", "Categorical vs Numeric", "Categorical vs Categorical"],
                      horizontal=True)

    if combo == "Numeric vs Numeric" and len(num_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X", num_cols, index=0, key="bi_x")
        y = c2.selectbox("Y", num_cols, index=1, key="bi_y")
        hue = c3.selectbox("Color by (optional)", ["None"] + cat_cols, key="bi_hue")
        fig = px.scatter(df, x=x, y=y, color=None if hue == "None" else hue,
                          color_discrete_sequence=COLORWAY, trendline="ols" if st.checkbox("Show trend line") else None)
        st.plotly_chart(style_fig(fig, f"{x} vs {y}", height=500), use_container_width=True)
        st.info(f"Pearson correlation (r): **{df[x].corr(df[y]):.3f}**")

        st.markdown("**Correlation heatmap — all numeric columns**")
        corr = df[num_cols].corr()
        fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(style_fig(fig2, "Correlation Heatmap", height=650), use_container_width=True)

    elif combo == "Categorical vs Numeric" and cat_cols and num_cols:
        c1, c2 = st.columns(2)
        cat = c1.selectbox("Categorical column", cat_cols, key="bi_cat")
        num = c2.selectbox("Numeric column", num_cols, key="bi_num")
        plot_kind = st.radio("Chart type", ["Box", "Violin", "Strip"], horizontal=True)
        if plot_kind == "Box":
            fig = px.box(df, x=cat, y=num, color=cat, color_discrete_sequence=COLORWAY, points="outliers")
        elif plot_kind == "Violin":
            fig = px.violin(df, x=cat, y=num, color=cat, color_discrete_sequence=COLORWAY, box=True)
        else:
            fig = px.strip(df, x=cat, y=num, color=cat, color_discrete_sequence=COLORWAY)
        st.plotly_chart(style_fig(fig, f"{num} by {cat}", height=500), use_container_width=True)
        st.dataframe(df.groupby(cat, observed=True)[num].describe(), use_container_width=True)

    elif combo == "Categorical vs Categorical" and len(cat_cols) >= 2:
        c1, c2 = st.columns(2)
        cat1 = c1.selectbox("Categorical column 1", cat_cols, index=0, key="bi_cat1")
        cat2 = c2.selectbox("Categorical column 2", cat_cols, index=min(1, len(cat_cols) - 1), key="bi_cat2")
        ct = pd.crosstab(df[cat1], df[cat2])
        fig = px.imshow(ct, text_auto=True, color_continuous_scale="Viridis", aspect="auto")
        st.plotly_chart(style_fig(fig, f"{cat1} x {cat2} — counts", height=500), use_container_width=True)
    else:
        st.info("Not enough columns of the required type are selected in the sidebar.")

# =======================================================================
# MULTIVARIATE
# =======================================================================
with tab_multi:
    st.markdown("### Multivariate Analysis — 3+ variables at once")

    st.markdown("####  Scatter Matrix")
    pair_cols = st.multiselect("Numeric columns", num_cols, default=num_cols[:4], key="matrix_cols")
    pair_hue = st.selectbox("Color by", ["None"] + cat_cols, key="matrix_hue")
    if len(pair_cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=pair_cols, color=None if pair_hue == "None" else pair_hue,
                                 color_discrete_sequence=COLORWAY)
        fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=5, opacity=0.7))
        st.plotly_chart(style_fig(fig, "Scatter Matrix", height=700), use_container_width=True)

    st.markdown("---")
    st.markdown("####  3D Scatter (x, y, z, color)")
    if len(num_cols) >= 3:
        c1, c2, c3, c4 = st.columns(4)
        zx = c1.selectbox("X", num_cols, index=0, key="3d_x")
        zy = c2.selectbox("Y", num_cols, index=1, key="3d_y")
        zz = c3.selectbox("Z", num_cols, index=2, key="3d_z")
        zc = c4.selectbox("Color", ["None"] + cat_cols, key="3d_c")
        fig = px.scatter_3d(df, x=zx, y=zy, z=zz, color=None if zc == "None" else zc,
                             color_discrete_sequence=COLORWAY, opacity=0.8)
        st.plotly_chart(style_fig(fig, f"{zx} · {zy} · {zz}", height=650), use_container_width=True)

    st.markdown("---")
    st.markdown("####  Parallel Coordinates")
    par_cols = st.multiselect("Numeric columns", num_cols, default=num_cols[:5], key="parallel_cols")
    if len(par_cols) >= 2:
        color_col = par_cols[0]
        fig = px.parallel_coordinates(df, dimensions=par_cols, color=color_col,
                                       color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(style_fig(fig, "Parallel Coordinates", height=500), use_container_width=True)

    st.markdown("---")
    st.markdown("####  Grouped Heatmap — mean of a numeric column across two categories")
    if len(cat_cols) >= 2 and num_cols:
        c1, c2, c3 = st.columns(3)
        g1 = c1.selectbox("Rows", cat_cols, index=0, key="multi_g1")
        g2 = c2.selectbox("Columns", cat_cols, index=min(1, len(cat_cols) - 1), key="multi_g2")
        val = c3.selectbox("Value", num_cols, key="multi_val")
        pivot = df.groupby([g1, g2], observed=True)[val].mean().unstack()
        fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="Magma", aspect="auto")
        st.plotly_chart(style_fig(fig, f"Mean {val} by {g1} x {g2}", height=500), use_container_width=True)

    st.markdown("---")
    st.markdown("####  Bubble Chart — 4 variables (x, y, size, color)")
    if len(num_cols) >= 3:
        c1, c2, c3, c4 = st.columns(4)
        bx = c1.selectbox("X", num_cols, index=0, key="bub_x")
        by = c2.selectbox("Y", num_cols, index=1, key="bub_y")
        bsize = c3.selectbox("Size", num_cols, index=2, key="bub_size")
        bhue = c4.selectbox("Color", ["None"] + cat_cols, key="bub_hue")
        fig = px.scatter(df, x=bx, y=by, size=bsize, color=None if bhue == "None" else bhue,
                          color_discrete_sequence=COLORWAY, size_max=45, opacity=0.75)
        st.plotly_chart(style_fig(fig, f"{bx} vs {by} (size = {bsize})", height=550), use_container_width=True)

# =======================================================================
# STATISTICAL TESTS
# =======================================================================
with tab_stats:
    st.markdown("### Statistical Tests — go beyond visuals to confirm what you see")

    st.markdown("####  Two-group comparison (independent t-test)")
    two_group_cats = [c for c in cat_cols if df[c].nunique() == 2]
    if two_group_cats and num_cols:
        c1, c2 = st.columns(2)
        tcat = c1.selectbox("Two-category column", two_group_cats, key="ttest_cat")
        tnum = c2.selectbox("Numeric column to compare", num_cols, key="ttest_num")
        groups = df[tcat].cat.categories if hasattr(df[tcat], "cat") else df[tcat].unique()
        g1_name, g2_name = list(df[tcat].unique())[:2]
        g1_vals = df.loc[df[tcat] == g1_name, tnum]
        g2_vals = df.loc[df[tcat] == g2_name, tnum]
        tstat, pval = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Mean ({g1_name})", f"{g1_vals.mean():.2f}")
        c2.metric(f"Mean ({g2_name})", f"{g2_vals.mean():.2f}")
        c3.metric("p-value", f"{pval:.4f}")
        if pval < 0.05:
            st.success(f" Statistically significant difference in **{tnum}** between {g1_name} and {g2_name} (p < 0.05).")
        else:
            st.info(f"No statistically significant difference in **{tnum}** between {g1_name} and {g2_name} (p ≥ 0.05).")
        fig = px.violin(df, x=tcat, y=tnum, color=tcat, box=True, points="all", color_discrete_sequence=COLORWAY)
        st.plotly_chart(style_fig(fig, f"{tnum} by {tcat}"), use_container_width=True)
    else:
        st.info("Need a categorical column with exactly 2 groups and at least one numeric column.")

    st.markdown("---")
    st.markdown("####  Multi-group comparison (one-way ANOVA)")
    multi_group_cats = [c for c in cat_cols if df[c].nunique() > 2]
    if multi_group_cats and num_cols:
        c1, c2 = st.columns(2)
        acat = c1.selectbox("Multi-category column", multi_group_cats, key="anova_cat")
        anum = c2.selectbox("Numeric column to compare", num_cols, key="anova_num")
        groups = [g[anum].values for _, g in df.groupby(acat, observed=True)]
        fstat, pval = stats.f_oneway(*groups)
        c1, c2 = st.columns(2)
        c1.metric("F-statistic", f"{fstat:.3f}")
        c2.metric("p-value", f"{pval:.4f}")
        if pval < 0.05:
            st.success(f" At least one group's mean **{anum}** differs significantly across **{acat}** (p < 0.05).")
        else:
            st.info(f"No statistically significant difference in **{anum}** across **{acat}** groups (p ≥ 0.05).")
        fig = px.box(df, x=acat, y=anum, color=acat, color_discrete_sequence=COLORWAY)
        st.plotly_chart(style_fig(fig, f"{anum} by {acat}"), use_container_width=True)
    else:
        st.info("Need a categorical column with more than 2 groups and at least one numeric column.")

    st.markdown("---")
    st.markdown("####  Correlation significance")
    if len(num_cols) >= 2:
        c1, c2 = st.columns(2)
        px_col = c1.selectbox("Variable 1", num_cols, index=0, key="pear_x")
        py_col = c2.selectbox("Variable 2", num_cols, index=1, key="pear_y")
        r, pval = stats.pearsonr(df[px_col], df[py_col])
        c1, c2 = st.columns(2)
        c1.metric("Pearson r", f"{r:.3f}")
        c2.metric("p-value", f"{pval:.4g}")
        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.3 else "weak"
        direction = "positive" if r > 0 else "negative"
        st.info(f"This is a **{strength} {direction}** correlation" +
                (", and it is statistically significant (p < 0.05)." if pval < 0.05 else ", but not statistically significant (p ≥ 0.05)."))

# =======================================================================
# INSIGHTS
# =======================================================================
with tab_insights:
    st.markdown("###  Auto-Generated Insights")

    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                 .stack().sort_values(ascending=False).head(5))
        st.markdown("**Top 5 strongest correlations**")
        for (f1, f2), val in pairs.items():
            st.markdown(f'<div class="insight-card"> <b>{f1}</b> and <b>{f2}</b> are correlated at '
                        f'<b>{val:.2f}</b></div>', unsafe_allow_html=True)

    if cat_cols and num_cols:
        st.markdown("**Group comparison — mean of each numeric column by category**")
        gcat = st.selectbox("Category", cat_cols, key="insight_cat")
        st.dataframe(
            df.groupby(gcat, observed=True)[num_cols].mean().round(2)
            .style.background_gradient(cmap="cool", axis=0),
            use_container_width=True,
        )

    st.markdown("---")
    st.download_button(
        "⬇️ Download cleaned data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
