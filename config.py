"""Configuration and constants for Enterprise Data Profiler Pro"""

import streamlit as st

# Page configuration
PAGE_CONFIG = {
    "page_title": "Enterprise Data Profiler Pro",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Enterprise CSS Styles
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Main Layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 15px;
        border-bottom: 2px solid #f1f5f9;
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Score Display */
    .score-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .score-label {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 8px;
    }
    
    /* Status Badges */
    .status-excellent { 
        background: #dcfce7; 
        color: #166534; 
        padding: 4px 12px; 
        border-radius: 9999px; 
        font-weight: 600;
        font-size: 0.875rem;
    }
    .status-good { 
        background: #dbeafe; 
        color: #1e40af; 
        padding: 4px 12px; 
        border-radius: 9999px; 
        font-weight: 600;
        font-size: 0.875rem;
    }
    .status-warning { 
        background: #fef3c7; 
        color: #92400e; 
        padding: 4px 12px; 
        border-radius: 9999px; 
        font-weight: 600;
        font-size: 0.875rem;
    }
    .status-critical { 
        background: #fee2e2; 
        color: #991b1b; 
        padding: 4px 12px; 
        border-radius: 9999px; 
        font-weight: 600;
        font-size: 0.875rem;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8fafc;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 6px;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1e40af !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1e293b;
        background: #f8fafc;
        border-radius: 8px;
        padding: 12px 16px;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 16px;
    }
    
    /* Workflow Steps */
    .workflow-step {
        text-align: center;
        padding: 16px;
        border-radius: 8px;
        background: #f1f5f9;
    }
    
    .workflow-step.active {
        background: #dbeafe;
        border: 2px solid #3b82f6;
    }
    
    .workflow-step.completed {
        background: #dcfce7;
        border: 2px solid #10b981;
    }
    
    /* Similarity Bar */
    .similarity-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .similarity-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Issue Cards */
    .issue-card {
        border-left: 4px solid;
        padding: 12px 16px;
        margin: 8px 0;
        background: #f8fafc;
        border-radius: 0 8px 8px 0;
    }
    
    .issue-card.critical { border-left-color: #ef4444; }
    .issue-card.warning { border-left-color: #f59e0b; }
    .issue-card.info { border-left-color: #3b82f6; }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
    }
    
    /* Toast notifications */
    .toast {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 9999;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #3b82f6;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 20px;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
</style>
"""


def apply_page_config():
    """Apply Streamlit page configuration"""
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)