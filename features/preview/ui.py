"""Preview Component - Performance-optimized data preview"""

import streamlit as st
import pandas as pd
import math

from state.session import st


def render_preview():
    """Performance-optimized data preview with pagination"""
    
    state = st.session_state.app_state
    
    if state.df is None:
        st.info("📤 No data loaded")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">👁️ Data Preview</div>', unsafe_allow_html=True)
    
    df = state.df
    
    # Dataset info
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.metric("Total Rows", f"{len(df):,}")
    with info_col2:
        st.metric("Total Columns", len(df.columns))
    with info_col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    with info_col4:
        # Data types summary
        dtype_counts = df.dtypes.value_counts()
        st.metric("Data Types", len(dtype_counts))
    
    # Pagination controls
    st.divider()
    
    PAGE_SIZES = [100, 500, 1000, 5000, 10000]
    
    pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 1, 1, 2])
    
    with pag_col1:
        page_size = st.selectbox("Rows per page:", PAGE_SIZES, index=0)
    
    total_pages = math.ceil(len(df) / page_size)
    
    with pag_col2:
        page_num = st.number_input("Page:", min_value=1, max_value=max(1, total_pages), value=1)
    
    with pag_col3:
        st.metric("Total Pages", total_pages)
    
    with pag_col4:
        # Quick navigation
        nav_cols = st.columns(4)
        with nav_cols[0]:
            if st.button("⏮️ First", width="stretch"):
                st.session_state.preview_page = 1
        with nav_cols[1]:
            if st.button("◀️ Prev", width="stretch") and page_num > 1:
                st.session_state.preview_page = page_num - 1
        with nav_cols[2]:
            if st.button("▶️ Next", width="stretch") and page_num < total_pages:
                st.session_state.preview_page = page_num + 1
        with nav_cols[3]:
            if st.button("⏭️ Last", width="stretch"):
                st.session_state.preview_page = total_pages
    
    # Calculate slice
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    
    # Column selection for performance
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        all_cols = df.columns.tolist()
        selected_cols = st.multiselect("Select columns to display:", 
                                      all_cols,
                                      default=all_cols[:20] if len(all_cols) > 20 else all_cols,
                                      key="preview_selected_cols")
    
    with col2:
        # Search within data
        search_term = st.text_input("🔍 Search in data:", "", key="preview_search")
    
    # Get data slice
    if selected_cols:
        display_df = df.iloc[start_idx:end_idx][selected_cols]
    else:
        display_df = df.iloc[start_idx:end_idx]
    
    # Apply search filter
    if search_term:
        mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False))
        display_df = display_df[mask.any(axis=1)]
        st.info(f"Search found {len(display_df)} matching rows")
    
    # Display with performance optimization
    st.markdown(f"**Showing rows {start_idx:,} to {end_idx:,} of {len(df):,}**")
    
    # Use st.dataframe with optimized settings
    st.dataframe(
        display_df,
        width="stretch",
        height=600,
        column_config={
            col: st.column_config.Column(
                col,
                help=f"Type: {df[col].dtype}",
                width="medium"
            ) for col in display_df.columns
        }
    )
    
    # Download current view
    st.divider()
    
    if st.button("📥 Download Current View as CSV", width="stretch"):
        csv = display_df.to_csv(index=False)
        st.download_button(
            "⬇️ Click to Download",
            data=csv,
            file_name=f"preview_rows_{start_idx}_to_{end_idx}.csv",
            mime="text/csv"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)