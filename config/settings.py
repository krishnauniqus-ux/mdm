"""Application settings and configuration"""

import streamlit as st
from .styles import CUSTOM_CSS

# Page configuration
PAGE_CONFIG = {
    "page_title": "Data Profiler Pro",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}


def apply_page_config():
    """Apply Streamlit page configuration and custom styles"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
