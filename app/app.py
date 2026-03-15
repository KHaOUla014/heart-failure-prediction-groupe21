import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
from datetime import datetime

sys.path.append('src')

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="CardioGuard",
    page_icon="❤️",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

    :root {
        --parchment:    #F5EFE6;
        --cream:        #FBF7F2;
        --crimson:      #8B1C1C;
        --crimson-soft: #A83232;
        --crimson-pale: #D4A8A8;
        --ink:          #2A1A1A;
        --ink-mid:      #4A3232;
        --ink-soft:     #7A5A5A;
        --border:       #DDD0C0;
        --shadow:       rgba(42,26,26,0.10);
    }

    html, body, [class*="css"] {
        font-family: 'Lato', sans-serif;
        background-color: var(--parchment) !important;
        color: var(--ink);
    }
    .main .block-container { background-color: var(--parchment) !important; max-width: 780px; }

    /* ── LANDING PAGE ── */
    .landing-wrap {
        min-height: 80vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 4rem 2rem;
    }
    .landing-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0.5rem 0 0;
        letter-spacing: 0.02em;
        line-height: 1.1;
    }
    .landing-sub {
        font-family: 'Lato', sans-serif;
        font-size: 1rem;
        font-weight: 300;
        color: var(--ink-soft);
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.75rem;
        margin-bottom: 2.5rem;
    }
    .landing-divider {
        width: 50px; height: 2px;
        background: var(--crimson);
        margin: 1.2rem auto;
        opacity: 0.6;
    }
    .landing-note {
        font-size: 0.78rem;
        color: var(--ink-soft);
        font-weight: 300;
        letter-spacing: 0.06em;
        margin-top: 1rem;
    }

    /* ── HERO (inner pages) ── */
    .hero {
        background: var(--cream);
        border-top: 4px solid var(--crimson);
        border: 1px solid var(--border);
        border-top: 4px solid var(--crimson);
        border-radius: 6px;
        padding: 2.2rem 2rem 1.6rem;
        margin-bottom: 1.8rem;
        text-align: center;
        box-shadow: 0 3px 18px var(--shadow);
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0.4rem 0 0;
        letter-spacing: 0.02em;
    }
    .hero-sub {
        font-size: 0.82rem;
        font-weight: 300;
        color: var(--ink-soft);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* ── HEARTBEAT ── */
    @keyframes hb       { 0%,100%{transform:scale(1)} 14%{transform:scale(1.15)} 28%{transform:scale(1)} 42%{transform:scale(1.10)} 56%{transform:scale(1)} }
    @keyframes hb-fast  { 0%,100%{transform:scale(1)} 10%{transform:scale(1.25)} 20%{transform:scale(1)} 30%{transform:scale(1.20)} 40%{transform:scale(1)} 50%{transform:scale(1.25)} 60%{transform:scale(1)} }
    @keyframes hb-calm  { 0%,100%{transform:scale(1)} 14%{transform:scale(1.08)} 28%{transform:scale(1)} 42%{transform:scale(1.05)} 56%{transform:scale(1)} }
    .heart-slow  { display:inline-block; animation:hb      2.5s ease-in-out infinite; filter:drop-shadow(0 2px 6px rgba(139,28,28,0.35)); }
    .heart-fast  { display:inline-block; animation:hb-fast 0.8s ease-in-out infinite; filter:drop-shadow(0 2px 10px rgba(139,28,28,0.6)); }
    .heart-calm  { display:inline-block; animation:hb-calm 4s  ease-in-out infinite; filter:drop-shadow(0 2px 6px rgba(60,140,80,0.4)); }

    /* ── STEP DOTS ── */
    .step-wrap { display:flex; justify-content:center; align-items:center; gap:0.5rem; margin:1rem 0 0.3rem; }
    .dot       { width:8px;height:8px;border-radius:50%;background:var(--border);border:1.5px solid var(--border);display:inline-block; }
    .dot-on    { background:var(--crimson);border-color:var(--crimson);box-shadow:0 0 5px rgba(139,28,28,0.4); }
    .dot-done  { background:var(--ink-soft);border-color:var(--ink-soft); }
    .step-lbl  { text-align:center;color:var(--ink-soft);font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1.6rem; }

    /* ── CARDS ── */
    .card {
        background: var(--cream);
        border: 1px solid var(--border);
        border-left: 3px solid var(--crimson);
        border-radius: 5px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px var(--shadow);
    }
    .card-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        color: var(--crimson);
        font-weight: 500;
        letter-spacing: 0.02em;
        margin-bottom: 0.1rem;
    }
    .card-desc { font-size:0.80rem; color:var(--ink-soft); font-weight:300; margin-bottom:0.9rem; }

    /* ── RISK RESULT ── */
    .risk-critical { background:#FAF0F0; border:1.5px solid #C44040; border-top:4px solid #C44040; border-radius:6px; padding:2rem; text-align:center; margin:1rem 0; box-shadow:0 4px 18px rgba(180,40,40,0.10); }
    .risk-high     { background:#FBF4EC; border:1.5px solid #C47030; border-top:4px solid #C47030; border-radius:6px; padding:2rem; text-align:center; margin:1rem 0; box-shadow:0 4px 18px rgba(180,100,40,0.10); }
    .risk-moderate { background:#FAFCEC; border:1.5px solid #B0A830; border-top:4px solid #B0A830; border-radius:6px; padding:2rem; text-align:center; margin:1rem 0; box-shadow:0 4px 18px rgba(160,150,40,0.08); }
    .risk-minimal  { background:#EDF8F0; border:1.5px solid #3A9A5A; border-top:4px solid #3A9A5A; border-radius:6px; padding:2rem; text-align:center; margin:1rem 0; box-shadow:0 4px 18px rgba(40,140,80,0.08); }

    .pct-critical { font-family:'Playfair Display',serif; font-size:5.5rem; color:#C02020; line-height:1; margin:0; font-weight:400; }
    .pct-high     { font-family:'Playfair Display',serif; font-size:5.5rem; color:#B86020; line-height:1; margin:0; font-weight:400; }
    .pct-moderate { font-family:'Playfair Display',serif; font-size:5.5rem; color:#908018; line-height:1; margin:0; font-weight:400; }
    .pct-minimal  { font-family:'Playfair Display',serif; font-size:5.5rem; color:#2A8A4A; line-height:1; margin:0; font-weight:400; }

    /* ── BUTTONS ── */
    .stButton > button {
        background: var(--crimson) !important;
        color: var(--cream) !important;
        border: 1.5px solid #6B1515 !important;
        border-radius: 4px !important;
        font-family: 'Lato', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: all 0.18s ease !important;
        padding: 0.5rem 1.4rem !important;
    }
    .stButton > button:hover {
        background: var(--crimson-soft) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 16px rgba(139,28,28,0.22) !important;
    }

    /* ── MISC ── */
    hr { border-color: var(--border) !important; }
    [data-testid="stSidebar"] { background: var(--cream) !important; border-right:1px solid var(--border) !important; }
    .stSlider label,.stSelectbox label,.stNumberInput label {
        font-family:'Lato',sans-serif !important;
        color:var(--ink-mid) !important;
        font-size:0.86rem !important;
        font-weight:700 !important;
        letter-spacing:0.05em !important;
    }
    .deco { display:flex; align-items:center; gap:0.7rem; margin:1.4rem 0 1rem; color:var(--crimson-pale); font-size:0.65rem; letter-spacing:0.18em; text-transform:uppercase; }
    .deco::before,.deco::after { content:''; flex:1; height:1px; background:var(--border); }

    /* download button */
    .stDownloadButton > button {
        background: var(--cream) !important;
        color: var(--crimson) !important;
        border: 1.5px solid var(--crimson) !important;
        border-radius: 4px !important;
        font-family:'Lato',sans-serif !important;
        font-size:0.85rem !important;
        font-weight:700 !important;
        letter-spacing:0.08em !important;
        text-transform:uppercase !important;
    }
    .stDownloadButton > button:hover {
        background: var(--crimson) !important;
        color: var(--cream) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 16px rgba(139,28,28,0.22) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# HELPERS
# ─────────────────────────────────────
def risk_zone(prob):
    if prob < 25:   return "minimal",  "#2A8A4A", "✅ Minimal Risk"
    elif prob < 50: return "moderate", "#908018", "⚠️ Moderate Risk"
    elif prob < 75: return "high",     "#B86020", "🔶 High Risk"
    else:           return "critical", "#C02020", "🔴 Critical Risk"


def draw_gauge(probability):
    fig, ax = plt.subplots(figsize=(5.5, 3), subplot_kw=dict(polar=False))
    fig.patch.set_facecolor('#FBF7F2'); ax.set_facecolor('#FBF7F2')
    ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
    zones = [
        (0,25,'#EDF8F0','#2A8A4A','Minimal'),
        (25,50,'#FAFCEC','#908018','Moderate'),
        (50,75,'#FBF4EC','#B86020','High'),
        (75,100,'#FAF0F0','#C02020','Critical'),
    ]
    for s,e,bg,col,lbl in zones:
        ts = np.pi*(1-s/100); te = np.pi*(1-e/100)
        th = np.linspace(ts,te,50)
        xo = 5+4.0*np.cos(th); yo = 1+4.0*np.sin(th)
        xi = 5+2.8*np.cos(th[::-1]); yi = 1+2.8*np.sin(th[::-1])
        ax.fill(np.concatenate([xo,xi]), np.concatenate([yo,yi]), color=bg, zorder=1)
        ax.plot(xo, yo, color=col, linewidth=2, zorder=2)
        mt = (ts+te)/2
        ax.text(5+3.5*np.cos(mt), 1+3.5*np.sin(mt), lbl,
                ha='center', va='center', fontsize=7, color=col,
                fontweight='bold', rotation=np.degrees(mt)-90)
    na = np.pi*(1-probability/100)
    ax.annotate('', xy=(5+2.5*np.cos(na),1+2.5*np.sin(na)), xytext=(5,1),
                arrowprops=dict(arrowstyle='->', color='#2A1818', lw=2.5))
    ax.plot(5,1,'o',color='#2A1818',markersize=8,zorder=10)
    _, col, _ = risk_zone(probability)
    ax.text(5,0.2,f'{probability:.0f}%',ha='center',va='center',
            fontsize=20,fontweight='bold',color=col,fontfamily='serif')
    plt.tight_layout(pad=0)
    return fig


def make_pdf(patient_data, probability, recs):
    try:
        from fpdf import FPDF
        def c(t): return str(t).encode('latin-1','ignore').decode('latin-1')
        zone, _, zlbl = risk_zone(probability)
        pdf = FPDF(); pdf.add_page()
        # header
        pdf.set_fill_color(139,28,28); pdf.rect(0,0,210,36,'F')
        pdf.set_font('Times','B',18); pdf.set_text_color(251,247,242)
        pdf.set_y(10); pdf.cell(0,10,c('CardioGuard — Clinical Heart Failure Report'),align='C')
        pdf.set_font('Times','',9); pdf.set_text_color(196,160,152)
        pdf.set_y(23); pdf.cell(0,7,c(f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}'),align='C')
        # risk
        pdf.set_y(44); pdf.set_font('Times','B',12); pdf.set_text_color(42,26,26)
        pdf.cell(0,8,c('RISK ASSESSMENT'),ln=True)
        cm = {'minimal':(42,138,74),'moderate':(144,128,24),'high':(184,96,32),'critical':(192,32,32)}
        r,g,b = cm.get(zone,(42,26,26))
        pdf.set_font('Times','B',24); pdf.set_text_color(r,g,b)
        clean_lbl = zlbl.replace("✅","").replace("⚠️","").replace("🔶","").replace("🔴","").strip()
        pdf.cell(0,12,c(f'{probability:.1f}%  —  {clean_lbl}'),ln=True)
        # patient data
        pdf.set_y(pdf.get_y()+4); pdf.set_font('Times','B',12); pdf.set_text_color(42,26,26)
        pdf.cell(0,8,c('PATIENT DATA'),ln=True)
        pdf.set_font('Times','',10); pdf.set_text_color(74,48,48)
        labels = {
            'age':'Age','anaemia':'Anaemia','creatinine_phosphokinase':'CPK (mcg/L)',
            'diabetes':'Diabetes','ejection_fraction':'Ejection Fraction (%)',
            'high_blood_pressure':'High Blood Pressure','platelets':'Platelets',
            'serum_creatinine':'Serum Creatinine (mg/dL)','serum_sodium':'Serum Sodium (mEq/L)',
            'sex':'Sex','smoking':'Smoking','time':'Follow-up (days)'
        }
        for col in patient_data.columns:
            val = patient_data[col].values[0]
            pdf.cell(95,7,c(labels.get(col,col)),border='B')
            pdf.cell(95,7,c(str(val)),border='B',ln=True)
        # interpretation
        pdf.set_y(pdf.get_y()+5); pdf.set_font('Times','B',12); pdf.set_text_color(42,26,26)
        pdf.cell(0,8,c('CLINICAL NOTES'),ln=True)
        pdf.set_font('Times','',10); pdf.set_text_color(74,48,48)
        note_map = {
            'critical': 'Patient presents with critical-level heart failure risk. Immediate specialist referral is strongly recommended.',
            'high':     'Patient presents with elevated risk. Prompt clinical follow-up and targeted intervention are advised.',
            'moderate': 'Patient presents with moderate risk. Regular monitoring and lifestyle modifications are recommended.',
            'minimal':  'Patient risk profile is currently minimal. Routine screening and preventive care should continue.'
        }
        pdf.multi_cell(0,6,c(note_map.get(zone,'')))
        return bytes(pdf.output())
    except ImportError:
        return None


# ─────────────────────────────────────
# SESSION INIT
# ─────────────────────────────────────
for k, v in [('page','landing'),('step',1),('model_loaded',False),
              ('patient_history',[]),('prediction_done',False)]:
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def load_model():
    from train_model import train_and_select_best_model
    model, X_train, X_test, y_train, y_test = train_and_select_best_model()
    return model, X_train, X_test, y_train, y_test


# ═══════════════════════════════════════
# PAGE 0 — LANDING
# ═══════════════════════════════════════
if st.session_state.page == 'landing':
    st.markdown("""
    <div class="landing-wrap">
        <span style="font-size:5rem; display:inline-block; animation:hb 2.5s ease-in-out infinite;
              filter:drop-shadow(0 3px 10px rgba(139,28,28,0.35));">❤️</span>
        <h1 class="landing-title">CardioGuard</h1>
        <div class="landing-divider"></div>
        <p class="landing-sub">Heart Failure · Clinical Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b:
        if st.button("🩺  Start the Examination", use_container_width=True):
            st.session_state.page = 'exam'
            st.rerun()

    st.markdown("""
    <p class="landing-note" style="text-align:center; margin-top:0.8rem;">
        🔒 &nbsp;Patient data is not stored · For clinical use only
    </p>
    """, unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════
# LOAD MODEL (once)
# ═══════════════════════════════════════
if not st.session_state.model_loaded:
    with st.spinner("⚕️ Loading..."):
        try:
            model, X_train, X_test, y_train, y_test = load_model()
            st.session_state.model_loaded = True
            st.session_state.model = model
        except Exception as e:
            st.error(f"❌ Could not load model: {e}"); st.stop()


# ═══════════════════════════════════════
# SIDEBAR — PATIENT HISTORY
# ═══════════════════════════════════════
if st.session_state.patient_history:
    with st.sidebar:
        st.markdown("""<p style="font-family:'Playfair Display',serif;font-size:1.1rem;
            color:#2A1A1A;margin-bottom:0.2rem;">📋 Patient History</p>
            <p style="font-size:0.75rem;color:#7A5A5A;font-weight:300;margin-bottom:0.6rem;">This session</p>
        """, unsafe_allow_html=True)
        for i, e in enumerate(reversed(st.session_state.patient_history)):
            _, col, lbl = risk_zone(e['probability'])
            st.markdown(f"""
            <div style="background:#FBF7F2;border:1px solid #DDD0C0;border-radius:4px;
                padding:0.65rem 0.9rem;margin:0.3rem 0;display:flex;
                justify-content:space-between;align-items:center;">
                <span style="color:#4A3232;font-size:0.80rem;">
                    #{len(st.session_state.patient_history)-i} · 🎂 {e['age']} · {e['sex']}
                </span>
                <span style="color:{col};font-weight:700;font-size:0.80rem;">{e['probability']:.0f}%</span>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑️ Clear"):
            st.session_state.patient_history = []; st.rerun()


# ═══════════════════════════════════════
# HERO
# ═══════════════════════════════════════
heart_cls = "heart-slow"
if st.session_state.prediction_done:
    p = st.session_state.get('last_prob', 50)
    heart_cls = "heart-fast" if p >= 75 else ("heart-calm" if p < 25 else "heart-slow")

st.markdown(f"""
<div class="hero">
    <span class="{heart_cls}" style="font-size:3.2rem;">❤️</span>
    <h1 class="hero-title">CardioGuard</h1>
    <p class="hero-sub">Heart Failure · Clinical Risk Assessment</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# STEP INDICATOR
# ═══════════════════════════════════════
labels = {1:"👤 Profile", 2:"🏥 Medical History", 3:"🔬 Lab Results", 4:"📊 Results"}
dots = ""
for i in range(1, 5):
    cls = "dot-done" if i < st.session_state.step else ("dot-on" if i == st.session_state.step else "dot")
    dots += f'<span class="dot {cls}"></span>'

st.markdown(f"""
<div class="step-wrap">{dots}</div>
<p class="step-lbl">Step {st.session_state.step} of 4 — {labels[st.session_state.step]}</p>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# STEP 1 — PROFILE
# ═══════════════════════════════════════
if st.session_state.step == 1:
    st.markdown("""<div class="card">
        <div class="card-title">👤 Patient Profile</div>
        <div class="card-desc">Basic demographic information about the patient</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age  = st.slider("🎂 Age", 40, 95, 60)
        sex  = st.selectbox("⚧ Sex", [0,1], format_func=lambda x: "👨 Male" if x==1 else "👩 Female")
    with c2:
        time    = st.slider("📅 Follow-up Period (days)", 4, 285, 130)
        smoking = st.selectbox("🚬 Smoking", [0,1], format_func=lambda x: "Yes — Smoker" if x==1 else "No — Non-smoker")

    st.session_state.s_age=age; st.session_state.s_sex=sex
    st.session_state.s_time=time; st.session_state.s_smoking=smoking

    if st.button("Continue → Step 2  ▶", use_container_width=True):
        st.session_state.step=2; st.rerun()


# ═══════════════════════════════════════
# STEP 2 — MEDICAL HISTORY
# ═══════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown("""<div class="card">
        <div class="card-title">🏥 Medical History</div>
        <div class="card-desc">Pre-existing conditions and chronic diagnoses</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        anaemia  = st.selectbox("🩸 Anaemia",  [0,1], format_func=lambda x: "Yes — Low red blood cells" if x==1 else "No")
        diabetes = st.selectbox("🍬 Diabetes", [0,1], format_func=lambda x: "Yes — Diabetic" if x==1 else "No")
    with c2:
        hbp = st.selectbox("💉 High Blood Pressure", [0,1], format_func=lambda x: "Yes — Hypertensive" if x==1 else "No")

    st.session_state.s_anaemia=anaemia; st.session_state.s_diabetes=diabetes; st.session_state.s_hbp=hbp

    c1,c2 = st.columns(2)
    with c1:
        if st.button("◀ Back", use_container_width=True): st.session_state.step=1; st.rerun()
    with c2:
        if st.button("Continue → Step 3  ▶", use_container_width=True): st.session_state.step=3; st.rerun()


# ═══════════════════════════════════════
# STEP 3 — LAB RESULTS
# ═══════════════════════════════════════
elif st.session_state.step == 3:
    st.markdown("""<div class="card">
        <div class="card-title">🔬 Clinical Measurements</div>
        <div class="card-desc">Laboratory values from the most recent assessment</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        ef  = st.slider("💓 Ejection Fraction (%)", 14, 80, 38,
                        help="% of blood pumped per beat. Normal: 55–70%. Below 40% = heart failure.")
        sc  = st.number_input("🧪 Serum Creatinine (mg/dL)", 0.5, 10.0, 1.2,
                              help="Kidney function marker. Normal: 0.6–1.2.")
        ss  = st.slider("🧂 Serum Sodium (mEq/L)", 100, 150, 137,
                        help="Normal: 135–145. Low sodium worsens prognosis.")
    with c2:
        cpk     = st.number_input("⚡ CPK (mcg/L)", 20, 8000, 250,
                                  help="Enzyme indicating heart/muscle damage.")
        plt_val = st.number_input("🩸 Platelets (kiloplatelets/mL)", 25000.0, 850000.0, 262000.0,
                                  help="Normal: 150,000–400,000.")

    st.session_state.s_ef=ef; st.session_state.s_sc=sc; st.session_state.s_ss=ss
    st.session_state.s_cpk=cpk; st.session_state.s_plt=plt_val

    c1,c2 = st.columns(2)
    with c1:
        if st.button("◀ Back", use_container_width=True): st.session_state.step=2; st.rerun()
    with c2:
        if st.button("🔍 Analyze Risk", use_container_width=True): st.session_state.step=4; st.rerun()


# ═══════════════════════════════════════
# STEP 4 — RESULTS
# ═══════════════════════════════════════
elif st.session_state.step == 4:

    s = st.session_state
    patient_data = pd.DataFrame([[
        s.s_age, s.s_anaemia, s.s_cpk, s.s_diabetes, s.s_ef,
        s.s_hbp, s.s_plt, s.s_sc, s.s_ss, s.s_sex, s.s_smoking, s.s_time
    ]], columns=[
        'age','anaemia','creatinine_phosphokinase','diabetes',
        'ejection_fraction','high_blood_pressure','platelets',
        'serum_creatinine','serum_sodium','sex','smoking','time'
    ])

    model       = s.model
    probability = model.predict_proba(patient_data)[0][1] * 100
    zone, zcol, zlbl = risk_zone(probability)

    s.last_prob = probability
    s.prediction_done = True
    sex_lbl = "Male" if s.s_sex == 1 else "Female"
    if not any(e['age']==s.s_age and abs(e['probability']-probability)<0.1
               for e in s.patient_history):
        s.patient_history.append({'age':s.s_age,'sex':sex_lbl,
            'probability':probability,'time':datetime.now().strftime("%H:%M")})

    hcls = "heart-fast" if zone=="critical" else ("heart-slow" if zone=="high" else "heart-calm")
    msg  = "⚠️ Immediate clinical attention recommended." if zone in ['critical','high'] \
           else "✅ Continue routine monitoring and preventive care."

    # ── RISK DISPLAY ──
    st.markdown(f"""
    <div class="risk-{zone}">
        <span class="{hcls}" style="font-size:3rem;">❤️</span>
        <p class="pct-{zone}">{probability:.0f}%</p>
        <p style="color:{zcol}; font-family:'Playfair Display',serif; font-size:1.1rem;
           font-weight:500; letter-spacing:0.08em; text-transform:uppercase; margin-top:0.4rem;">
            {zlbl}
        </p>
        <p style="color:#7A5A5A; font-size:0.85rem; margin-top:0.5rem; font-weight:300;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── GAUGE ──
    st.markdown('<div class="deco">🎯 Risk Gauge</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">
        <div class="card-title">🎯 Risk Gauge</div>
        <div class="card-desc">Visual representation of the patient's position across risk zones</div>
    </div>""", unsafe_allow_html=True)
    fig = draw_gauge(probability); st.pyplot(fig, use_container_width=False); plt.close()

    # ── SHAP ──
    st.markdown('<div class="deco">🔬 Key Factors</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">
        <div class="card-title">🔬 Key Clinical Factors</div>
        <div class="card-desc">
            🔴 Red bars = factors that increased the risk &nbsp;·&nbsp;
            🔵 Blue bars = factors that reduced the risk &nbsp;·&nbsp;
            Longer bar = stronger influence
        </div>
    </div>""", unsafe_allow_html=True)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(patient_data)
        fig2, _ = plt.subplots(figsize=(8,4))
        fig2.patch.set_facecolor('#FBF7F2')
        if len(sv.shape)==3: shap.plots.waterfall(sv[0,:,1], show=False)
        else:                shap.plots.waterfall(sv[0], show=False)
        plt.tight_layout(); st.pyplot(fig2); plt.close()
    except Exception as e:
        st.warning(f"Factor analysis unavailable: {e}")

    # ── PATIENT SUMMARY ──
    st.markdown('<div class="deco">📋 Patient Summary</div>', unsafe_allow_html=True)
    with st.expander("📋 View Full Patient Data", expanded=False):
        display = patient_data.rename(columns={
            'age':'🎂 Age','anaemia':'🩸 Anaemia','creatinine_phosphokinase':'⚡ CPK',
            'diabetes':'🍬 Diabetes','ejection_fraction':'💓 Ejection Fraction (%)',
            'high_blood_pressure':'💉 High BP','platelets':'🩸 Platelets',
            'serum_creatinine':'🧪 Serum Creatinine','serum_sodium':'🧂 Serum Sodium',
            'sex':'⚧ Sex','smoking':'🚬 Smoking','time':'📅 Follow-up (days)'
        })
        st.dataframe(display, use_container_width=True)

    # ── PDF DOWNLOAD ──
    st.markdown('<div class="deco">📄 Export Report</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card">
        <div class="card-title">📄 Download Clinical Report</div>
        <div class="card-desc">Save a complete PDF report for the patient's medical record</div>
    </div>""", unsafe_allow_html=True)

    pdf_bytes = make_pdf(patient_data, probability, [])
    if pdf_bytes:
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name=f"cardioguard_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.info("💡 Install `fpdf2` to enable PDF: `pip install fpdf2`")
        csv = patient_data.copy()
        csv['risk_%'] = probability; csv['zone'] = zlbl
        st.download_button(
            label="⬇️  Download CSV Report",
            data=csv.to_csv(index=False),
            file_name=f"cardioguard_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── NEW PATIENT ──
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄  New Patient", use_container_width=True):
            st.session_state.step=1; st.session_state.prediction_done=False; st.rerun()
    with c2:
        if st.button("🏠  Back to Home", use_container_width=True):
            st.session_state.page='landing'; st.session_state.step=1
            st.session_state.prediction_done=False; st.rerun()