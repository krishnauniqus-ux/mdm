"""
Enterprise Data Profiler Pro
Production-grade data quality platform with large file support
"""

import sys
import warnings

# Suppress WebSocket errors BEFORE importing streamlit
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*WebSocket.*')
warnings.filterwarnings('ignore', message='.*StreamClosedError.*')

# Configure logging first
import logging
logging.getLogger('tornado').setLevel(logging.CRITICAL)
logging.getLogger('streamlit').setLevel(logging.WARNING)

# Patch WebSocket handler
try:
    from utils.websocket_handler import patch_streamlit_websocket, configure_tornado_logging
    configure_tornado_logging()
    patch_streamlit_websocket()
except Exception:
    pass  # Continue even if patching fails

import streamlit as st

# Must be first Streamlit command
st.set_page_config(
    page_title="Profiler Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- IMPORT COMPONENTS --------------------

from ui import render_header, render_sidebar
from features.load_data import render_load_data
from features.profiling import render_data_profiling
from features.rule_generator import render_rule_generator
from features.duplicates import render_find_duplicates
from features.quality import render_data_quality
from features.compare import render_compare
from features.preview import render_preview
from features.export import render_export

from state.session import init_session_state, render_toasts as session_render_toasts


def main():
    """Main application with sequential tab workflow"""

    # Initialize session state
    init_session_state()

    # Render layout
    render_header()

    # Authentication Guard
    if not st.session_state.app_state.authenticated:
        from auth.ui import render_login_screen
        render_login_screen()
        return

    render_sidebar()

    # Render toast notifications
    session_render_toasts()

    state = st.session_state.app_state

    # -------------------- TABS --------------------

    tabs = st.tabs([
        "1️⃣ Load Data",
        "2️⃣ Rule Generator",
        "3️⃣ Data Profiling",
        "4️⃣ Find Duplicates",
        "5️⃣ Data Quality",
        "6️⃣ Compare",
        "7️⃣ Preview",
        "8️⃣ Export"
    ])

    # -------------------- TAB 1: Load Data --------------------
    with tabs[0]:
        render_load_data()

    # -------------------- TAB 2: Rule Generator --------------------
    with tabs[1]:
        if state.df is not None:
            render_rule_generator()
        else:
            st.info("⏳ Complete 'Load Data' step first")

    # -------------------- TAB 3: Data Profiling --------------------
    with tabs[2]:
        if state.df is not None:
            render_data_profiling()
        else:
            st.info("⏳ Complete 'Load Data' step first")

    # -------------------- TAB 4: Find Duplicates --------------------
    with tabs[3]:
        if state.df is not None:
            render_find_duplicates()
        else:
            st.info("⏳ Complete 'Load Data' step first")

    # -------------------- TAB 5: Data Quality --------------------
    with tabs[4]:
        if state.df is not None:
            render_data_quality()
        else:
            st.info("⏳ Complete 'Load Data' step first")

    # -------------------- TAB 6: Compare --------------------
    with tabs[5]:
        if state.df is not None and state.original_df is not None:
            render_compare()
        else:
            st.info("⏳ Load data to enable comparison")

    # -------------------- TAB 7: Preview --------------------
    with tabs[6]:
        if state.df is not None:
            render_preview()
        else:
            st.info("⏳ Complete 'Load Data' step first")

    # -------------------- TAB 8: Export --------------------
    with tabs[7]:
        if state.df is not None:
            render_export()
        else:
            st.info("⏳ Complete 'Load Data' step first")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Graceful handling of WebSocket-related issues
        if "WebSocket" in str(e) or "StreamClosed" in str(e):
            st.error("Connection interrupted. Please refresh the page.")
            st.stop()
        else:
            raise
