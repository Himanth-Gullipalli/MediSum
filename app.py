"""
app.py – MediSum: AI-Powered Medical Report Summarization System
Main Streamlit application.

Navigation (sidebar):
  🏥  Dashboard        – Stats and recent activity
  👤  Patient Management – Add / search patients
  📄  Upload & Analyze  – Upload PDFs, run RAG, generate summary
  📋  Summary History   – Browse past summaries per patient
  ⚙️  Settings          – API key status, DB connection info
"""

import os
import time
import logging
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from auth import is_logged_in, render_login_page, logout
from database import MediSumDB
from rag_pipeline import process_reports
from pdf_generator import create_summary_pdf

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediSum – AI Medical Report Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ══════════════════════════════════════════
       SIDEBAR  –  deep navy, bright white text
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background-color: #1e2d4f !important;
        border-right: 2px solid #2d4070;
    }
    /* Force all sidebar text to bright white */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #f1f5f9 !important;
    }
    /* Radio button labels */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        padding: 0.25rem 0 !important;
    }
    /* Selected radio dot */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {
        border-color: #60a5fa !important;
        background-color: #60a5fa !important;
    }
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #2d4070 !important;
        color: #f1f5f9 !important;
        border: 1px solid #3d5494 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #3d5494 !important;
    }

    /* ══════════════════════════════════════════
       MAIN AREA  –  clean off-white background
    ══════════════════════════════════════════ */
    .stApp { background-color: #f8fafc !important; }
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* All regular text in main area: dark charcoal */
    .main p, .main span, .main div, .main label {
        color: #1e293b;
    }

    /* ══════════════════════════════════════════
       METRIC CARDS
    ══════════════════════════════════════════ */
    [data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-top: 4px solid #2563eb !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    /* Metric label – dark grey, clearly visible */
    [data-testid="metric-container"] label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Metric value – dark navy, large */
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1e2d4f !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* ══════════════════════════════════════════
       CARDS  –  white with blue left accent
    ══════════════════════════════════════════ */
    .medisum-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.2rem;
    }
    .medisum-card h3 {
        color: #1e2d4f !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.7rem !important;
    }
    .medisum-card p, .medisum-card li, .medisum-card span {
        color: #334155 !important;
        font-size: 0.93rem !important;
        line-height: 1.65 !important;
    }

    /* ══════════════════════════════════════════
       SECTION HEADER  –  white bg, navy text
       (NO dark gradient – easy to read)
    ══════════════════════════════════════════ */
    .section-header {
        background: #ffffff;
        color: #1e2d4f !important;
        border-left: 6px solid #2563eb;
        border-bottom: 1px solid #e2e8f0;
        padding: 0.85rem 1.2rem;
        border-radius: 0 10px 10px 0;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* ══════════════════════════════════════════
       SUMMARY BOX  –  white, readable
    ══════════════════════════════════════════ */
    .summary-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #16a34a;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        font-size: 0.95rem;
        line-height: 1.75;
        color: #1e293b !important;
    }
    .summary-box p, .summary-box li, .summary-box strong {
        color: #1e293b !important;
    }
    .summary-box strong { font-weight: 700 !important; }
    .summary-box h1, .summary-box h2, .summary-box h3 {
        color: #1e2d4f !important;
        font-weight: 700 !important;
    }

    /* ══════════════════════════════════════════
       PATIENT BADGE  –  light mint, dark text
    ══════════════════════════════════════════ */
    .patient-badge {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1rem;
    }
    .patient-badge .name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #14532d !important;
    }
    .patient-badge .details {
        color: #166534 !important;
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }

    /* ══════════════════════════════════════════
       BUTTONS  –  solid navy primary
    ══════════════════════════════════════════ */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.18s !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #f1f5f9 !important;
        border-color: #2563eb !important;
        color: #2563eb !important;
    }

    /* ══════════════════════════════════════════
       FILE UPLOADER
    ══════════════════════════════════════════ */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #2563eb !important;
        border-radius: 10px !important;
        background: #eff6ff !important;
    }

    /* ══════════════════════════════════════════
       PROGRESS BAR  –  blue
    ══════════════════════════════════════════ */
    [data-testid="stProgress"] > div > div {
        background-color: #2563eb !important;
    }

    /* ══════════════════════════════════════════
       ALERTS / INFO BOXES
    ══════════════════════════════════════════ */
    .stAlert { border-radius: 10px !important; }

    /* ══════════════════════════════════════════
       EXPANDERS
    ══════════════════════════════════════════ */
    [data-testid="stExpander"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        overflow: hidden;
        background: #ffffff !important;
    }
    [data-testid="stExpander"] summary {
        color: #1e2d4f !important;
        font-weight: 600 !important;
        background: #f8fafc !important;
    }

    /* ══════════════════════════════════════════
       TABS
    ══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
        background: #eff6ff !important;
    }

    /* ══════════════════════════════════════════
       FORM INPUTS  –  clear borders
    ══════════════════════════════════════════ */
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        border-color: #cbd5e1 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
        background: #ffffff !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
    }
    .stTextInput label, .stTextArea label, .stSelectbox label,
    .stNumberInput label, .stRadio label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.87rem !important;
    }

    /* ══════════════════════════════════════════
       SIDEBAR LOGO BLOCK
    ══════════════════════════════════════════ */
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 0.8rem;
        border-bottom: 1px solid #2d4070;
        margin-bottom: 1rem;
    }
    .sidebar-logo .app-name {
        font-size: 1.5rem;
        font-weight: 800;
        color: #93c5fd !important;
        letter-spacing: 1px;
    }
    .sidebar-logo .app-tagline {
        font-size: 0.72rem;
        color: #94a3b8 !important;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Database Singleton ────────────────────────────────────────────────────────
@st.cache_resource
def get_db() -> MediSumDB:
    return MediSumDB()


# ── Auth Guard ────────────────────────────────────────────────────────────────
if not is_logged_in():
    render_login_page()
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <div style="font-size:2.2rem;">🏥</div>
            <div class="app-name">MediSum</div>
            <div class="app-tagline">AI Medical Report Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏥  Dashboard",
            "👤  Patient Management",
            "📄  Upload & Analyze",
            "📋  Summary History",
            "⚙️  Settings",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Current patient badge
    if st.session_state.get("current_patient"):
        pt = st.session_state["current_patient"]
        st.markdown(
            f"""
            <div style="background:rgba(0,172,193,0.1); border-radius:10px; 
                        padding:0.7rem; font-size:0.82rem; margin-bottom:0.5rem;">
                <b style="color:#00acc1;">Active Patient</b><br>
                <span style="color:#ecf0f1;">{pt['name']}</span><br>
                <span style="color:#90a4ae; font-size:0.75rem;">
                    Age {pt.get('age','—')} · {pt.get('gender','—')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    user = st.session_state.get("current_user", "user")
    if st.button(f"🚪  Logout ({user})", use_container_width=True):
        logout()
        st.rerun()


# ── Helper ────────────────────────────────────────────────────────────────────
def section_header(icon: str, title: str):
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏥  Dashboard":
    db = get_db()
    stats = db.get_stats()

    section_header("🏥", "Dashboard")

    # ── Stats row ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Patients",   stats.get("patients",  0))
    c2.metric("📝 Total Summaries",  stats.get("summaries", 0))
    c3.metric("🤖 LLM",             "Groq / Llama 3")
    c4.metric("💾 Vector Store",     "ChromaDB")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="medisum-card"><h3>📖 How MediSum Works</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            1. **Add a Patient** — Register a patient in the system with their demographic details.
            2. **Upload Reports** — Upload one or more PDF medical reports (doctor notes, scan reports, blood tests, etc.).
            3. **AI Analysis** — MediSum extracts text, builds a semantic vector store with ChromaDB, retrieves the most relevant passages, and feeds them to the Groq LLM to generate a structured summary.
            4. **Download & Store** — Download the summary as a styled PDF and save it to MongoDB for future reference.
            5. **History** — Browse all past summaries for any patient at any time.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="medisum-card"><h3>⚡ Quick Actions</h3>', unsafe_allow_html=True)
        if st.button("➕  Add New Patient",   use_container_width=True, type="primary"):
            st.session_state["_nav"] = "👤  Patient Management"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📄  Upload & Analyze",  use_container_width=True):
            st.session_state["_nav"] = "📄  Upload & Analyze"
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋  View Summaries",    use_container_width=True):
            st.session_state["_nav"] = "📋  Summary History"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # DB status
        if stats.get("connected"):
            st.success("✅ MongoDB connected")
        else:
            st.warning("⚠️ MongoDB not connected – data will not be saved. Check your `.env` / secrets.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Patient Management
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤  Patient Management":
    db = get_db()
    section_header("👤", "Patient Management")

    tab_add, tab_search, tab_all = st.tabs(["➕ Add Patient", "🔍 Search Patient", "📋 All Patients"])

    # ── Tab: Add Patient ──────────────────────────────────────
    with tab_add:
        st.markdown("#### Register a New Patient")
        with st.form("add_patient_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            name       = c1.text_input("Full Name *",          placeholder="e.g. Ravi Kumar")
            age        = c2.number_input("Age *",              min_value=0, max_value=120, value=30)
            gender     = c1.selectbox("Gender *",             ["Male", "Female", "Other", "Prefer not to say"])
            blood_group= c2.selectbox("Blood Group",          ["—", "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
            contact    = c1.text_input("Contact / Phone",      placeholder="e.g. +91 98765 43210")
            notes      = st.text_area("Additional Notes",     placeholder="Pre-existing conditions, allergies, etc.", height=100)

            submitted = st.form_submit_button("💾  Register Patient", type="primary", use_container_width=True)

        if submitted:
            if not name.strip():
                st.error("Patient name is required.")
            else:
                pid = db.create_patient(
                    name=name,
                    age=int(age),
                    gender=gender,
                    contact=contact,
                    blood_group=blood_group if blood_group != "—" else "",
                    notes=notes,
                )
                if pid:
                    patient = db.get_patient(pid)
                    st.session_state["current_patient"] = patient
                    st.success(f"✅ Patient **{name}** registered successfully! (ID: `{pid}`)")
                    st.info("💡 This patient is now set as the active patient. Head to **Upload & Analyze** to process their reports.")
                else:
                    st.warning("⚠️ Patient data saved in session only (MongoDB not connected).")
                    st.session_state["current_patient"] = {
                        "_id": "demo-id",
                        "name": name, "age": age, "gender": gender,
                        "blood_group": blood_group, "contact": contact,
                    }

    # ── Tab: Search Patient ───────────────────────────────────
    with tab_search:
        st.markdown("#### Search Existing Patient")
        query = st.text_input("Search by name or contact", placeholder="Type to search…")

        if query:
            results = db.search_patients(query)
            if results:
                for pt in results:
                    col_info, col_btn = st.columns([5, 1])
                    with col_info:
                        st.markdown(
                            f"""
                            <div class="patient-badge">
                                <div class="name">{pt['name']}</div>
                                <div class="details">
                                    Age: {pt.get('age','—')} &nbsp;|&nbsp; 
                                    Gender: {pt.get('gender','—')} &nbsp;|&nbsp; 
                                    Blood: {pt.get('blood_group','—')} &nbsp;|&nbsp; 
                                    Contact: {pt.get('contact','—')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_btn:
                        if st.button("Select", key=f"sel_{pt['_id']}"):
                            st.session_state["current_patient"] = pt
                            st.success(f"✅ Active patient set to **{pt['name']}**")
                            st.rerun()
            else:
                st.info("No patients found matching your search.")

    # ── Tab: All Patients ─────────────────────────────────────
    with tab_all:
        st.markdown("#### All Registered Patients")
        patients = db.list_all_patients()

        if not patients:
            st.info("No patients registered yet. Use the **Add Patient** tab to get started.")
        else:
            for pt in patients:
                col_info, col_sel, col_del = st.columns([5, 1, 1])
                with col_info:
                    st.markdown(
                        f"""
                        <div class="patient-badge">
                            <div class="name">{pt['name']}</div>
                            <div class="details">
                                Age: {pt.get('age','—')} &nbsp;|&nbsp; 
                                Gender: {pt.get('gender','—')} &nbsp;|&nbsp; 
                                Blood: {pt.get('blood_group','—')}
                                {' &nbsp;|&nbsp; ' + pt['contact'] if pt.get('contact') else ''}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_sel:
                    if st.button("✅ Select", key=f"all_sel_{pt['_id']}"):
                        st.session_state["current_patient"] = pt
                        st.success(f"Active patient: **{pt['name']}**")
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{pt['_id']}", help="Delete patient"):
                        if db.delete_patient(pt["_id"]):
                            if st.session_state.get("current_patient", {}).get("_id") == pt["_id"]:
                                del st.session_state["current_patient"]
                            st.success(f"Patient {pt['name']} deleted.")
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Upload & Analyze
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄  Upload & Analyze":
    db = get_db()
    section_header("📄", "Upload & Analyze Reports")

    # ── Patient selector ──────────────────────────────────────
    current_patient = st.session_state.get("current_patient")

    if not current_patient:
        patients = db.list_all_patients()
        if patients:
            st.markdown("#### Select a Patient")
            options = {f"{p['name']} (Age {p.get('age','?')})": p for p in patients}
            chosen = st.selectbox("Choose patient", list(options.keys()))
            if st.button("Set as Active Patient", type="primary"):
                st.session_state["current_patient"] = options[chosen]
                current_patient = options[chosen]
                st.rerun()
            st.stop()
        else:
            st.warning("⚠️ No patients found. Please add a patient first in **Patient Management**.")
            st.stop()

    # ── Active patient card ───────────────────────────────────
    pt = current_patient
    st.markdown(
        f"""
        <div class="patient-badge">
            <div class="name">🧑‍⚕️ {pt['name']}</div>
            <div class="details">
                Age: {pt.get('age','—')} &nbsp;|&nbsp; 
                Gender: {pt.get('gender','—')} &nbsp;|&nbsp; 
                Blood Group: {pt.get('blood_group','—')} &nbsp;|&nbsp; 
                Contact: {pt.get('contact','—') or '—'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_change, _ = st.columns([2, 5])
    with col_change:
        if st.button("🔄 Change Patient"):
            del st.session_state["current_patient"]
            if "current_summary" in st.session_state:
                del st.session_state["current_summary"]
            st.rerun()

    st.markdown("---")

    # ── File uploader ─────────────────────────────────────────
    st.markdown("#### 📂 Upload Medical Report PDFs")
    st.markdown(
        "_You can upload multiple PDFs at once (doctor reports, scan reports, blood reports, etc.)._"
    )

    tab_upload, tab_types = st.tabs(["📤 Upload Files", "ℹ️ Report Type Guide"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Drop PDF files here or click to browse",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF medical reports to analyze together.",
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) ready for analysis.")
            for uf in uploaded_files:
                st.markdown(f"- 📄 `{uf.name}` ({uf.size / 1024:.1f} KB)")

    with tab_types:
        st.markdown(
            """
            | Report Type | What to Upload |
            |---|---|
            | 🩺 Doctor Report | Discharge summaries, clinical notes, GP letters |
            | 🔬 Blood Report | CBC, metabolic panel, HbA1c, lipid profile |
            | 🖼️ Scan Report | MRI/CT/X-Ray radiologist reports (text PDF) |
            | 💊 Prescription | Medication lists, treatment plans |
            | 🧪 Lab Report | Urine analysis, culture reports, biopsy |
            
            > **Note:** PDFs must contain **selectable text**. Scanned image-only PDFs will not be readable.
            """
        )

    st.markdown("---")

    # ── Analyze button ────────────────────────────────────────
    if uploaded_files:
        col_btn, col_info = st.columns([2, 5])
        with col_btn:
            analyze_clicked = st.button(
                "🤖  Generate AI Summary",
                type="primary",
                use_container_width=True,
            )

        if analyze_clicked:
            progress_bar  = st.progress(0)
            status_text   = st.empty()

            def update_progress(step: int, total: int, msg: str):
                progress_bar.progress(step / total)
                status_text.markdown(f"**{msg}**")

            with st.spinner("Analyzing your medical reports…"):
                summary = process_reports(
                    uploaded_files=uploaded_files,
                    patient_info={
                        "name":         pt.get("name", "Unknown"),
                        "age":          pt.get("age",  "—"),
                        "gender":       pt.get("gender", "—"),
                        "blood_group":  pt.get("blood_group", "—"),
                        "contact":      pt.get("contact", "—"),
                        "patient_id":   pt.get("_id", "—"),
                        "report_types": [f.name for f in uploaded_files],
                    },
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0)
            status_text.empty()

            st.session_state["current_summary"]       = summary
            st.session_state["current_report_files"]  = [f.name for f in uploaded_files]

    # ── Display Summary ───────────────────────────────────────
    if st.session_state.get("current_summary"):
        summary = st.session_state["current_summary"]

        st.markdown("---")
        st.markdown("### 📋 AI-Generated Summary")
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown(summary)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        col_dl, col_save, _ = st.columns([2, 2, 3])

        # ── Download PDF ──────────────────────────────────────
        with col_dl:
            patient_info_for_pdf = {
                "name":         pt.get("name", "Unknown"),
                "age":          pt.get("age",  "—"),
                "gender":       pt.get("gender", "—"),
                "blood_group":  pt.get("blood_group", "—"),
                "contact":      pt.get("contact", "—"),
                "patient_id":   pt.get("_id", "—"),
                "report_types": st.session_state.get("current_report_files", []),
            }
            try:
                pdf_bytes = create_summary_pdf(patient_info_for_pdf, summary)
                st.download_button(
                    label="📥  Download PDF Summary",
                    data=pdf_bytes,
                    file_name=f"medisum_{pt.get('name','patient').replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        # ── Save to MongoDB ───────────────────────────────────
        with col_save:
            if st.button("💾  Save Summary to History", use_container_width=True):
                patient_id = pt.get("_id")
                if patient_id and patient_id != "demo-id":
                    pdf_bytes_to_save = None
                    try:
                        pdf_bytes_to_save = create_summary_pdf(patient_info_for_pdf, summary)
                    except Exception:
                        pass

                    sid = db.save_summary(
                        patient_id=patient_id,
                        summary_text=summary,
                        report_types=st.session_state.get("current_report_files", []),
                        pdf_bytes=pdf_bytes_to_save,
                    )
                    if sid:
                        st.success(f"✅ Summary saved! (ID: `{sid[:12]}…`)")
                    else:
                        st.warning("⚠️ Could not save — MongoDB not connected.")
                else:
                    st.warning("⚠️ Connect MongoDB to save summaries.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Summary History
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Summary History":
    db = get_db()
    section_header("📋", "Summary History")

    # ── Patient selector ──────────────────────────────────────
    patients = db.list_all_patients()
    if not patients:
        st.info("No patients found. Register a patient and generate summaries first.")
        st.stop()

    current_patient = st.session_state.get("current_patient")
    patient_names   = [f"{p['name']} (Age {p.get('age','?')})" for p in patients]
    patient_map     = {f"{p['name']} (Age {p.get('age','?')})": p for p in patients}

    default_idx = 0
    if current_patient:
        for i, p in enumerate(patients):
            if p["_id"] == current_patient.get("_id"):
                default_idx = i
                break

    selected_label = st.selectbox("Select Patient", patient_names, index=default_idx)
    selected_pt    = patient_map[selected_label]

    st.markdown("---")

    summaries = db.get_summaries(selected_pt["_id"])

    if not summaries:
        st.info(f"No summaries found for **{selected_pt['name']}**. Upload reports and generate a summary first.")
    else:
        st.markdown(f"**{len(summaries)} summary/summaries** found for **{selected_pt['name']}**")

        for idx, sm in enumerate(summaries):
            created = sm.get("created_at")
            if hasattr(created, "strftime"):
                ts_str = created.strftime("%d %b %Y, %H:%M")
            else:
                ts_str = str(created)[:16]

            report_types = sm.get("report_types", [])
            label = f"📝 Summary {idx+1} — {ts_str}"
            if report_types:
                label += f"  ({', '.join(report_types[:3])})"

            with st.expander(label, expanded=(idx == 0)):
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.markdown(sm.get("summary_text", "_No summary text_"))
                st.markdown("</div>", unsafe_allow_html=True)

                col_dl, _ = st.columns([2, 5])
                with col_dl:
                    # Try to fetch stored PDF
                    pdf_b = db.get_summary_pdf(sm["_id"])
                    if pdf_b:
                        st.download_button(
                            label="📥  Download PDF",
                            data=pdf_b,
                            file_name=f"medisum_{selected_pt['name'].replace(' ','_')}_{ts_str.replace(' ','_').replace(',','')}.pdf",
                            mime="application/pdf",
                            key=f"dl_{sm['_id']}",
                        )
                    else:
                        # Re-generate PDF on-the-fly
                        try:
                            regen_pdf = create_summary_pdf(
                                {
                                    "name":         selected_pt["name"],
                                    "age":          selected_pt.get("age", "—"),
                                    "gender":       selected_pt.get("gender", "—"),
                                    "blood_group":  selected_pt.get("blood_group", "—"),
                                    "contact":      selected_pt.get("contact", "—"),
                                    "patient_id":   selected_pt["_id"],
                                    "report_types": report_types,
                                },
                                sm.get("summary_text", ""),
                            )
                            st.download_button(
                                label="📥  Download PDF (regenerated)",
                                data=regen_pdf,
                                file_name=f"medisum_{selected_pt['name'].replace(' ','_')}.pdf",
                                mime="application/pdf",
                                key=f"regen_{sm['_id']}",
                            )
                        except Exception:
                            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Settings
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Settings":
    db = get_db()
    section_header("⚙️", "Settings & Configuration")

    col_l, col_r = st.columns(2)

    with col_l:
        # ── API Key status ────────────────────────────────────
        st.markdown('<div class="medisum-card"><h3>🤖 Groq API</h3>', unsafe_allow_html=True)
        try:
            groq_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
        except Exception:
            groq_key = os.getenv("GROQ_API_KEY", "")

        if groq_key:
            masked = groq_key[:6] + "••••••••" + groq_key[-4:]
            st.success(f"✅ Groq API key configured: `{masked}`")
        else:
            st.error("❌ GROQ_API_KEY not set.")
            st.markdown(
                """
                **To fix this:**
                1. Sign up at [console.groq.com](https://console.groq.com) (free)
                2. Create an API key
                3. Add it to your `.env` file:
                   ```
                   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
                   ```
                4. Or add it to Streamlit secrets (for cloud deployment)
                """
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── MongoDB status ────────────────────────────────────
        st.markdown('<div class="medisum-card"><h3>💾 MongoDB</h3>', unsafe_allow_html=True)
        stats = db.get_stats()
        if stats.get("connected"):
            st.success(f"✅ MongoDB connected — {stats['patients']} patients, {stats['summaries']} summaries")
        else:
            st.error("❌ MongoDB not connected.")
            st.markdown(
                """
                **To fix this:**
                - **Local**: Install MongoDB and start it, then set:
                  ```
                  MONGO_URI=mongodb://localhost:27017/
                  ```
                - **Cloud (free)**: Create an [Atlas](https://www.mongodb.com/atlas) free tier cluster and use:
                  ```
                  MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
                  ```
                """
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        # ── Model info ────────────────────────────────────────
        st.markdown('<div class="medisum-card"><h3>🔧 Model Configuration</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            | Setting | Value |
            |---|---|
            | LLM Model | `llama3-8b-8192` (Groq) |
            | Embedding Model | `all-MiniLM-L6-v2` (local) |
            | Chunk Size | 1,000 tokens |
            | Chunk Overlap | 200 tokens |
            | Retrieved Chunks | 6 per query |
            | Vector Store | ChromaDB (in-memory) |
            | Max Summary Tokens | 2,048 |
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Deployment guide ──────────────────────────────────
        st.markdown('<div class="medisum-card"><h3>🚀 Streamlit Cloud Deployment</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            1. Push this project to a **GitHub repository**
            2. Go to [share.streamlit.io](https://share.streamlit.io)
            3. Connect your repo and set `app.py` as the main file
            4. Add these **Secrets** in the Streamlit Cloud dashboard:
            ```toml
            GROQ_API_KEY = "gsk_..."
            MONGO_URI    = "mongodb+srv://..."
            MONGO_DB_NAME = "medisum"
            APP_USERNAME = "admin"
            APP_PASSWORD = "your_password"
            ```
            5. Deploy! 🎉
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
