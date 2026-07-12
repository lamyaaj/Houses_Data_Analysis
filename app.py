import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import streamlit.components.v1 as components

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="House Price Prediction Dashboard 🏡",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600&display=swap');

    :root {
        --page-bg: #F4F6F8;
        --card-color: #FFFFFF;
        --border-soft: #D8DDE3;
        --text-color: #1F2937;
        --text-muted: #4B5563;
        --hero-start: #6D5BE0;
        --hero-mid: #7B5FE8;
        --hero-end: #9B7BEF;
        --header-start: #232733;
        --header-mid: #34415C;
        --header-end: #4F72A6;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--text-color);
    }
    h1, h2, h3, h4 {
        font-family: 'Poppins', 'Inter', sans-serif;
    }

    .stApp {
        background: var(--page-bg);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---------- Hero Header ---------- */
    .app-header {
        padding: 2.6rem 3rem;
        background: linear-gradient(120deg, var(--header-start) 0%, var(--header-mid) 55%, var(--header-end) 100%);
        border-radius: 26px;
        margin-bottom: 2rem;
        box-shadow: 0 18px 40px rgba(35, 39, 51, 0.35);
        animation: fadeIn 0.6s ease;
        text-align: center;
    }
    .app-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .app-header h3 {
        font-size: 1.2rem;
        font-weight: 600;
        color: #D7E4F5;
        margin: 0 0 1.3rem 0;
    }
    .app-header .hero-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 999px;
        padding: 0.9rem 1.8rem;
        font-size: 0.98rem;
        color: #EAF1FB;
        line-height: 1.6;
        max-width: 880px;
        backdrop-filter: blur(6px);
    }

    /* ---------- Section titles ---------- */
    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 1.6rem 0 1.1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---------- KPI Cards ---------- */
    .kpi-card {
        border-radius: 20px;
        padding: 1.7rem 1.6rem;
        box-shadow: 0 12px 28px rgba(31, 36, 48, 0.14);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        text-align: left;
        height: 100%;
        color: #FFFFFF;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 36px rgba(31, 36, 48, 0.22);
    }
    .kpi-card.grad-1 { background: linear-gradient(135deg, #F857A6 0%, #FF5E62 100%); }
    .kpi-card.grad-2 { background: linear-gradient(135deg, #36D1DC 0%, #5B86E5 100%); }
    .kpi-card.grad-3 { background: linear-gradient(135deg, #43E97B 0%, #38F9D7 100%); }

    .kpi-icon-label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 1.1rem;
        opacity: 0.97;
    }
    .kpi-value {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.18);
    }

    /* ---------- Chart cards ---------- */
    .chart-card {
        background: var(--card-color);
        border: 1px solid var(--border-soft);
        border-radius: 20px;
        padding: 1.3rem 1.4rem 0.6rem 1.4rem;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.10);
        margin-bottom: 1.6rem;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }
    .chart-card:hover {
        box-shadow: 0 14px 30px rgba(31, 36, 48, 0.12);
        transform: translateY(-2px);
    }
    .chart-card h4 {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 0.1rem 0 0.8rem 0.1rem;
    }

    /* ---------- Predictor form card ---------- */
    .form-section {
        background: var(--card-color);
        border: 1px solid var(--border-soft);
        border-radius: 20px;
        padding: 1.7rem 1.8rem;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.10);
        margin-bottom: 1.3rem;
    }
    .form-section h4 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6D5BE0;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ---------- Result banner ---------- */
    .result-banner {
        background: linear-gradient(120deg, var(--hero-start) 0%, var(--hero-mid) 50%, var(--hero-end) 100%);
        border-radius: 22px;
        padding: 2rem 2rem;
        text-align: center;
        color: #FFFFFF;
        margin-top: 1.2rem;
        box-shadow: 0 18px 40px rgba(109, 91, 224, 0.35);
        animation: popIn 0.45s ease;
    }
    .result-banner .label {
        font-size: 1rem;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 0.4rem;
    }
    .result-banner .value {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(120deg, #6D5BE0 0%, #9B7BEF 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1.8rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.25s ease;
        box-shadow: 0 10px 22px rgba(109, 91, 224, 0.35);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(109, 91, 224, 0.5);
        filter: brightness(1.05);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] h2 {
        font-size: 1.15rem;
        font-weight: 800;
        color: #6D5BE0;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        color: var(--text-color);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: #F7F8FC;
        border-radius: 12px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.4rem;
        transition: all 0.2s ease;
        border: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: #EFEBFF;
        border-color: #9B7BEF;
    }
    .sidebar-info-box {
        background: linear-gradient(135deg, #F0EEFD 0%, #F7F8FC 100%);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.7;
    }

    /* ---------- Inputs (soft modern restyle, never dark) ---------- */
    .stNumberInput input,
    .stTextInput input,
    div[data-baseweb="select"] > div {
        background-color: #F6F5FD !important;
        border-radius: 12px !important;
        border: 1.5px solid #E4E0FA !important;
        color: var(--text-color) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    .stNumberInput input:hover,
    .stTextInput input:hover,
    div[data-baseweb="select"] > div:hover {
        border-color: #C9C1F5 !important;
    }
    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: #6D5BE0 !important;
        box-shadow: 0 0 0 3px rgba(109, 91, 224, 0.15) !important;
        outline: none !important;
    }
    .stNumberInput button {
        background-color: #EFEBFF !important;
        border-color: #E4E0FA !important;
    }
    .stNumberInput button svg {
        fill: #6D5BE0 !important;
    }
    div[data-baseweb="select"] {
        border-radius: 12px !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #6D5BE0 !important;
    }

    /* ---------- Divider ---------- */
    .soft-divider {
        border: none;
        border-top: 1px solid var(--border-soft);
        margin: 1.8rem 0;
    }

    /* ---------- Key Insights ---------- */
    .insight-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7F5FF 100%);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.08);
        height: 100%;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        border-left: 5px solid #6D5BE0;
    }
    .insight-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 26px rgba(109, 91, 224, 0.18);
    }
    .insight-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .insight-title {
        font-size: 1rem;
        font-weight: 800;
        color: var(--text-color);
        margin-bottom: 0.4rem;
        line-height: 1.3;
    }
    .insight-text {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--text-muted);
        line-height: 1.55;
    }

    /* ---------- Footer Card ---------- */
    .footer-card {
        margin-top: 2.2rem;
        text-align: center;
        background: linear-gradient(120deg, #EFEBFF 0%, #E6F3FF 50%, #FDEFF6 100%);
        border: 1px solid var(--border-soft);
        border-radius: 22px;
        padding: 1.8rem 2rem;
        box-shadow: 0 12px 30px rgba(109, 91, 224, 0.14);
    }
    .footer-card .footer-icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
        display: block;
    }
    .footer-card .footer-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 0.2rem;
    }
    .footer-card .footer-name {
        font-size: 1.3rem;
        font-weight: 800;
        color: #6D5BE0;
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.01em;
    }
    .footer-card .footer-year {
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* ---------- Animations ---------- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.94); }
        to { opacity: 1; transform: scale(1); }
    }

    /* ---------- Contrast fixes for native Streamlit widgets ---------- */
    label, .stMarkdown p, .stMarkdown li,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }
    .stSlider label, .stNumberInput label, .stSelectbox label, .stRadio label {
        color: var(--text-color) !important;
    }
    .stRadio div[role="radiogroup"] label p {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: var(--text-muted);
    }
    .stAlert p {
        color: var(--text-color) !important;
        font-weight: 500;
    }

    /* ---------- Responsive tweaks ---------- */
    @media (max-width: 768px) {
        .app-header h1 { font-size: 1.8rem; }
        .kpi-value { font-size: 1.5rem; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Force English (Latin) numerals on native number inputs, regardless of the
# browser/OS locale, so the dashboard stays fully English end-to-end.
components.html("""
<script>
try {
    const doc = window.parent.document;
    doc.documentElement.setAttribute('lang', 'en');
    function forceEnglishNumerals() {
        doc.querySelectorAll('input[type="number"], input[type="text"]').forEach(function (inp) {
            inp.setAttribute('lang', 'en');
            inp.style.direction = 'ltr';
            inp.style.textAlign = 'left';
        });
    }
    forceEnglishNumerals();
    const observer = new MutationObserver(forceEnglishNumerals);
    observer.observe(doc.body, { childList: true, subtree: true });
} catch (e) {}
</script>
""", height=0, width=0)

# ============================================================
# 3. DATA LOADING & MODEL TRAINING
#    (UNCHANGED — identical ML logic and preprocessing)
# ============================================================
@st.cache_resource
def initialize_engine():
    # قراءة الملف المنظف
    df = pd.read_csv('clean_data.csv')

    # نسخة للعمليات لحماية البيانات الأصلية المستخدمة في الداشبورد
    df_cleaned = df.copy()

    # 1. حذف الأعمدة غير المفيدة كما فعلتِ في كودكِ تماماً
    columns_to_drop = ['street', 'statezip', 'country', 'date']
    existing_drops = [col for col in columns_to_drop if col in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(existing_drops, axis=1)

    # 2. تحويل عمود المدينة من نص إلى أرقام
    le = LabelEncoder()
    if 'city' in df_cleaned.columns:
        df_cleaned['city_encoded'] = le.fit_transform(df_cleaned['city'])
        df_cleaned = df_cleaned.drop('city', axis=1)

    # 3. إضافة عمود log_price إذا ما كان موجود
    if 'log_price' not in df_cleaned.columns and 'price' in df_cleaned.columns:
        df_cleaned['log_price'] = np.log1p(df_cleaned['price'])

    # 4. تجهيز المتغيرات للتدريب (X و y)
    drop_cols = [col for col in ['price', 'log_price'] if col in df_cleaned.columns]
    X = df_cleaned.drop(drop_cols, axis=1)
    y = df_cleaned['log_price']

    # حفظ ترتيب الأعمدة الدقيق لـ واجهة المستخدم
    features_order = list(X.columns)

    # 5. تدريب نموذج XGBoost بالإعدادات الفائزة حقتكِ بالملّي 🎯
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)

    return model, le, df, features_order


# تشغيل المحرك وسحب البيانات والترتيب الصحيح للأعمدة
model, le, df_clean, features_order = initialize_engine()

# ============================================================
# 3B. Shared chart styling helpers (visual polish only)
# ============================================================
BASE_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#1F2937", size=15),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(font=dict(color="#1F2937", size=13)),
)

def style_axes(fig, x_title=None, y_title=None, x_tickangle=0):
    fig.update_xaxes(
        title_text=x_title if x_title is not None else fig.layout.xaxis.title.text,
        tickfont=dict(color="#1F2937", size=14),
        title_font=dict(color="#1F2937", size=16),
        showgrid=False,
        tickangle=x_tickangle,
    )
    fig.update_yaxes(
        title_text=y_title if y_title is not None else fig.layout.yaxis.title.text,
        tickfont=dict(color="#1F2937", size=14),
        title_font=dict(color="#1F2937", size=16),
        gridcolor="#E5E7EB",
    )
    return fig

# ============================================================
# 4. HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>🏡 House Price Prediction Dashboard</h1>
    <h3>📊 Housing Market Analysis</h3>
    <div class="hero-pill">
        📈 Explore residential property data, including location, house size, number of bedrooms
        and bathrooms, property condition, and house prices. This dashboard helps users understand
        housing market trends and estimate property values through an interactive experience.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📊 Housing Analysis", "🧮 Smart Price Estimator"],
    label_visibility="collapsed"
)
st.sidebar.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='sidebar-info-box'>"
    "🏠 This dashboard uses an <b>XGBoost</b> regression model trained on Seattle-area "
    "housing data to estimate fair market value and surface key market trends."
    "</div>",
    unsafe_allow_html=True
)

# ============================================================
# 6A. PAGE — SMART PRICE ESTIMATOR (Predictor)
# ============================================================
if page == "🧮 Smart Price Estimator":
    st.markdown('<div class="section-title">🔮 Property Price Predictor</div>', unsafe_allow_html=True)
    st.info("Enter the property specifications below to receive an instant estimated market value.")

    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown("<h4>🏗️ Structural Details</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input("🛏️ Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("🛁 Bathrooms", 1.0, 8.0, 2.0, 0.5)
        floors = st.number_input("🏢 Floors", 1.0, 4.0, 1.0, 0.5)

    with col2:
        sqft_living = st.number_input("📐 Living Area (sqft)", 300, 15000, 2000)
        sqft_lot = st.number_input("🌳 Lot Size (sqft)", 500, 100000, 5000)
        condition = st.slider("🔧 House Condition", 1, 5, 3)

    with col3:
        city = st.selectbox("📍 City", sorted(df_clean['city'].unique()))
        waterfront = st.radio("🌊 Waterfront View?", ["No", "Yes"], horizontal=True)
        view = st.slider("🖼️ View Quality", 0, 4, 0)
    st.markdown('</div>', unsafe_allow_html=True)

    # معالجة المدخلات المباشرة لتطابق تدريب النموذج (UNCHANGED)
    waterfront_val = 1 if waterfront == "Yes" else 0
    city_encoded = le.transform([city])[0]

    # بناء قاموس يحتوي على كافة الميزات المحتملة وتعبئتها ديناميكياً (UNCHANGED)
    raw_user_inputs = {
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
        'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': waterfront_val,
        'view': view, 'condition': condition, 'city_encoded': city_encoded
    }

    # دعم الميزات الإضافية (UNCHANGED)
    for col in features_order:
        if col not in raw_user_inputs:
            if 'above' in col:
                raw_user_inputs[col] = sqft_living
            elif 'built' in col:
                raw_user_inputs[col] = 1980
            else:
                raw_user_inputs[col] = 0

    # إعادة ترتيب المدخلات (UNCHANGED)
    input_array = [raw_user_inputs[col] for col in features_order]

    center_col = st.columns([1, 1.4, 1])[1]
    with center_col:
        predict_clicked = st.button("🚀 Calculate Estimated Price")

    if predict_clicked:
        prediction_log = model.predict([input_array])
        prediction_actual = np.expm1(prediction_log)[0]

        st.markdown(f"""
        <div class="result-banner">
            <div class="label">✨ Estimated Market Value</div>
            <div class="value">${prediction_actual:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
        st.toast("Prediction complete! 🎉", icon="✅")

# ============================================================
# 6B. PAGE — HOUSING ANALYSIS (Dashboard)
# ============================================================
else:
    st.markdown('<div class="section-title">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown(f"""
        <div class="kpi-card grad-1">
            <div class="kpi-icon-label">🏠 Total Properties</div>
            <div class="kpi-value">{len(df_clean):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card grad-2">
            <div class="kpi-icon-label">💰 Average House Price</div>
            <div class="kpi-value">${df_clean['price'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card grad-3">
            <div class="kpi-icon-label">📐 Average Living Area</div>
            <div class="kpi-value">{df_clean['sqft_living'].mean():,.0f} sqft</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Market Visualizations</div>', unsafe_allow_html=True)

    COLORFUL_PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Vivid

    row2_col1, row2_col2 = st.columns(2)

    # ---------------- Chart 1: Average House Price by City ----------------
    with row2_col1:
        st.markdown('<div class="chart-card"><h4>🏙️ Average House Price by City</h4>', unsafe_allow_html=True)
        city_prices = (
            df_clean.groupby('city')['price']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        city_prices['price_label'] = city_prices['price'].apply(lambda v: f"${v:,.0f}")

        fig_city = px.bar(
            city_prices, x='city', y='price', color='city',
            color_discrete_sequence=COLORFUL_PALETTE,
            text='price_label',
            labels={'price': 'Average Price', 'city': 'City'}
        )
        fig_city.update_traces(
            textposition='outside',
            textfont=dict(size=13, color="#1F2937"),
            cliponaxis=False,
        )
        fig_city.update_layout(**BASE_LAYOUT, showlegend=False)
        style_axes(fig_city, x_title="City", y_title="Average Price", x_tickangle=-45)
        fig_city.update_yaxes(range=[0, city_prices['price'].max() * 1.18])
        st.plotly_chart(fig_city, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Chart 2: Living Area vs House Price ----------------
    with row2_col2:
        st.markdown('<div class="chart-card"><h4>📏 Living Area vs House Price</h4>', unsafe_allow_html=True)
        sample_df = df_clean.sample(1000) if len(df_clean) > 1000 else df_clean
        fig_scatter = px.scatter(
            sample_df, x='sqft_living', y='price', color='condition',
            opacity=0.75,
            color_continuous_scale=['#9B7BEF', '#6D5BE0', '#F857A6'],
            labels={'sqft_living': 'Living Area (sqft)', 'price': 'Price ($)', 'condition': 'Condition'}
        )
        fig_scatter.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#FFFFFF")))
        fig_scatter.update_layout(**BASE_LAYOUT)
        style_axes(fig_scatter, x_title="Living Area (sqft)", y_title="Price ($)")
        fig_scatter.update_layout(
            coloraxis_colorbar=dict(title=dict(text="Condition", font=dict(color="#1F2937", size=13)),
                                     tickfont=dict(color="#1F2937", size=12))
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row3_col1, row3_col2 = st.columns(2)

    # ---------------- Chart 3: House Condition Distribution ----------------
    with row3_col1:
        st.markdown('<div class="chart-card"><h4>🏠 House Condition Distribution</h4>', unsafe_allow_html=True)
        cond_counts = df_clean['condition'].value_counts().reset_index()
        cond_counts.columns = ['condition', 'count']
        cond_counts['condition_label'] = 'Condition ' + cond_counts['condition'].astype(str)
        fig_pie = px.pie(
            cond_counts, values='count', names='condition_label', hole=0.55,
            color_discrete_sequence=['#F857A6', '#36D1DC', '#43E97B', '#9B7BEF', '#FFA451']
        )
        fig_pie.update_traces(
            textposition='outside',
            textinfo='label+percent',
            texttemplate='%{label}<br>%{percent}',
            textfont=dict(size=11, color="#1F2937"),
            marker=dict(line=dict(color="#FFFFFF", width=2)),
            pull=[0.03] * len(cond_counts),
            rotation=15,
        )
        fig_pie.update_layout(
            **BASE_LAYOUT,
            showlegend=False,
            uniformtext_minsize=10,
            uniformtext_mode='show',
        )
        fig_pie.update_layout(margin=dict(l=60, r=60, t=30, b=30))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Chart 4: Average House Price by Number of Bedrooms (Treemap) ----------------
    with row3_col2:
        st.markdown('<div class="chart-card"><h4>🛏️ Average House Price by Number of Bedrooms</h4>', unsafe_allow_html=True)
        bedroom_prices = (
            df_clean[df_clean['bedrooms'] != 0]
            .groupby('bedrooms')['price']
            .mean()
            .reset_index()
            .sort_values('bedrooms')
        )
        bedroom_prices['bedrooms_label'] = bedroom_prices['bedrooms'].astype(int).astype(str) + " Bedrooms"
        bedroom_prices['price_label'] = bedroom_prices['price'].apply(lambda v: f"${v:,.0f}")
        bedroom_prices['display'] = bedroom_prices['bedrooms_label'] + "<br>" + bedroom_prices['price_label']

        fig_tree = px.treemap(
            bedroom_prices, path=['display'], values='price', color='bedrooms_label',
            color_discrete_sequence=COLORFUL_PALETTE,
            hover_data={'bedrooms_label': True, 'price_label': True, 'display': False}
        )
        fig_tree.update_traces(
            texttemplate='<b>%{label}</b>',
            textfont=dict(size=15, color="#FFFFFF"),
            marker=dict(line=dict(color="#FFFFFF", width=2)),
            root_color="rgba(0,0,0,0)",
        )
        fig_tree.update_layout(
            **BASE_LAYOUT,
            showlegend=False,
        )
        fig_tree.update_layout(margin=dict(l=4, r=4, t=10, b=4))
        st.plotly_chart(fig_tree, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # KEY INSIGHTS
    # ============================================================
    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Key Insights</div>', unsafe_allow_html=True)

    insights = [
        ("📊", "Significant Price Variation",
         "House prices vary considerably across the market, indicating a clear distinction "
         "between premium and mid-range residential areas."),
        ("📐", "Living Area Strongly Influences Price",
         "Larger living spaces generally command higher selling prices, although location and "
         "property condition also play an important role in determining value."),
        ("🏠", "Most Properties Share Similar Condition Ratings",
         "The majority of properties fall within the same condition category, suggesting a "
         "relatively consistent housing quality across the market, while price differences are "
         "mainly driven by other factors."),
        ("💎", "Premium Pricing Is Not Driven by Size Alone",
         "Some properties achieve exceptionally high prices despite having similar living areas "
         "to others, highlighting the influence of factors such as location, waterfront views, "
         "and overall property quality."),
    ]

    ins_col1, ins_col2, ins_col3, ins_col4 = st.columns(4)
    for col, (icon, title, text) in zip([ins_col1, ins_col2, ins_col3, ins_col4], insights):
        with col:
            st.markdown(f"""
            <div class="insight-card">
                <span class="insight-icon">{icon}</span>
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# 7. FOOTER CARD
# ============================================================
st.markdown("""
<div class="footer-card">
    <span class="footer-icon">✨</span>
    <div class="footer-label">Designed &amp; Developed by</div>
    <div class="footer-name">Lamyaa Aljohani✨</div>
    <div class="footer-year">© 2026</div>
</div>
""", unsafe_allow_html=True)