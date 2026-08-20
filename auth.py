"""
auth.py – Simple login/logout management for MediSum.

Uses Streamlit session_state to track login status.
Credentials are loaded from environment variables / Streamlit secrets.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_credentials() -> tuple[str, str]:
    """
    Return (username, password) from Streamlit secrets (cloud)
    or environment variables (local).
    """
    try:
        username = st.secrets["APP_USERNAME"]
        password = st.secrets["APP_PASSWORD"]
    except (FileNotFoundError, KeyError):
        username = os.getenv("APP_USERNAME", "admin")
        password = os.getenv("APP_PASSWORD", "medisum123")
    return username, password


def is_logged_in() -> bool:
    """Return True if the user is currently authenticated."""
    return st.session_state.get("logged_in", False)


def login(username: str, password: str) -> bool:
    """
    Validate credentials and set session state on success.
    Returns True on successful login.
    """
    valid_user, valid_pass = _get_credentials()
    if username == valid_user and password == valid_pass:
        st.session_state["logged_in"] = True
        st.session_state["current_user"] = username
        return True
    return False


def logout() -> None:
    """Clear all session state and log the user out."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def render_login_page() -> None:
    """Render a styled login card. Blocks until user logs in."""
    # ── Custom CSS ───────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Hide default streamlit chrome on login page */
        #MainMenu, footer, header {visibility: hidden;}

        .login-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 80vh;
        }
        .login-card {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border-radius: 20px;
            padding: 3rem 2.5rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            max-width: 420px;
            margin: auto;
            text-align: center;
        }
        .login-logo {
            font-size: 3rem;
            margin-bottom: 0.25rem;
        }
        .login-title {
            color: #e0f7fa;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 1px;
            margin-bottom: 0.25rem;
        }
        .login-subtitle {
            color: #80cbc4;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        .stTextInput > div > div > input {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 10px !important;
            color: white !important;
            padding: 0.75rem 1rem !important;
        }
        .stTextInput label {
            color: #b2dfdb !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stForm"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Center the card ──────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class="login-card">
                <div class="login-logo">🏥</div>
                <div class="login-title">MediSum</div>
                <div class="login-subtitle">AI-Powered Medical Report Analysis</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )
            submitted = st.form_submit_button(
                "🔐  Sign In", use_container_width=True, type="primary"
            )

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
            elif login(username, password):
                st.success("Login successful! Loading dashboard…")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

        st.markdown(
            "<p style='color:#546e7a; font-size:0.75rem; text-align:center; margin-top:1rem;'>"
            "Default: admin / medisum123 (change in .env)</p>",
            unsafe_allow_html=True,
        )
