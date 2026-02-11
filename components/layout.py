"""Layout and UI components"""

import streamlit as st
from config import apply_page_config


def render_header():
    """Render application header"""
    apply_page_config()
    st.markdown('<h1 class="main-header">🔍 Enterprise Data Profiler Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Production-Grade Data Quality Platform • Supports files up to 1 GB</p>', unsafe_allow_html=True)


def render_sidebar():
    """Sidebar with navigation and stats"""
    
    state = st.session_state.app_state
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=80)
        st.title("Navigation")
        
        if state.df is not None:
            # Quick stats
            st.markdown("### 📊 Dataset Stats")
            st.metric("Rows", f"{len(state.df):,}")
            st.metric("Columns", len(state.df.columns))
            
            if state.quality_report:
                score = state.quality_report.overall_score
                color = "normal" if score >= 75 else "off" if score >= 60 else "inverse"
                st.metric("Quality Score", f"{score:.0f}/100", delta_color=color)
            
            # Processing status
            status_colors = {
                'idle': '⚪',
                'uploading': '🟡',
                'profiling': '🔵',
                'ready': '🟢',
                'error': '🔴'
            }
            st.markdown(f"**Status:** {status_colors.get(state.processing_status, '⚪')} {state.processing_status.title()}")
            
            # Operations count
            if state.fixes_applied:
                st.metric("Operations", len(state.fixes_applied))
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Reset Application", width="stretch", type="secondary"):
                from state.session import reset_application
                reset_application()
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        **Enterprise Features:**
        • 📤 Up to 1 GB file support
        • 🔍 Advanced data profiling
        • 📝 Regex & business rules
        • ⚡ Real-time preview
        • 📊 Export reports
        """)
        
        # System info
        st.markdown("---")
        st.caption(f"Version: 2.0.0 Enterprise")
        st.caption(f"Session ID: {id(st.session_state) % 10000}")


def render_toasts():
    """Placeholder - toasts are handled by session module"""
    pass