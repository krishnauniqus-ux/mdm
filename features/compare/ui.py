"""Compare Component - Simple Side-by-Side Data Comparison"""

import streamlit as st
import pandas as pd
from typing import List

from state.session import st as session_st


def render_compare():
    """Simple side-by-side comparison with highlighted changes"""
    
    state = session_st.session_state.app_state
    
    if state.df is None or state.original_df is None:
        st.info("📤 Load data and make changes to use comparison")
        return
    
    # Custom CSS for clean comparison
    st.markdown("""
    <style>
    .compare-container {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    
    .data-panel {
        flex: 1;
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .original-panel {
        border-left: 4px solid #3b82f6;
    }
    
    .modified-panel {
        border-left: 4px solid #10b981;
    }
    
    .panel-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
        padding: 15px;
        background: #f9fafb;
        border-radius: 8px;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
    }
    
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f2937;
        margin-top: 5px;
    }
    
    .change-positive {
        color: #10b981;
    }
    
    .change-negative {
        color: #ef4444;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get dataframes
    orig_df = state.original_df
    curr_df = state.df
    
    # Header
    st.markdown("## ⚖️ Compare: Original vs Modified")
    
    # Quick stats
    row_change = len(curr_df) - len(orig_df)
    col_change = len(curr_df.columns) - len(orig_df.columns)
    
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-label">Original Rows</div>
            <div class="stat-value">{len(orig_df):,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Modified Rows</div>
            <div class="stat-value">{len(curr_df):,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Row Change</div>
            <div class="stat-value {'change-positive' if row_change >= 0 else 'change-negative'}">{row_change:+,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Original Columns</div>
            <div class="stat-value">{len(orig_df.columns)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Modified Columns</div>
            <div class="stat-value">{len(curr_df.columns)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Column Change</div>
            <div class="stat-value {'change-positive' if col_change >= 0 else 'change-negative'}">{col_change:+}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Column selection
    orig_cols = set(orig_df.columns)
    curr_cols = set(curr_df.columns)
    common_cols = list(orig_cols & curr_cols)
    
    if not common_cols:
        st.warning("No common columns found between original and modified data")
        return
    
    # Settings
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_cols = st.multiselect(
            "Select columns to compare:",
            sorted(common_cols),
            default=sorted(common_cols)[:10] if len(common_cols) >= 10 else sorted(common_cols),
            key="compare_cols"
        )
    
    with col2:
        start_row = st.number_input(
            "Start row:",
            min_value=0,
            max_value=max(len(orig_df), len(curr_df)) - 1,
            value=0,
            key="start_row"
        )
    
    with col3:
        num_rows = st.number_input(
            "Rows to show:",
            min_value=1,
            max_value=500,
            value=50,
            key="num_rows"
        )
    
    if not selected_cols:
        st.warning("Please select at least one column to compare")
        return
    
    st.divider()
    
    # Side-by-side comparison
    end_row = start_row + num_rows
    
    # Prepare data with change detection
    comparison_data = _prepare_comparison_data(orig_df, curr_df, selected_cols, start_row, end_row)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="data-panel original-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📘 ORIGINAL DATA</div>', unsafe_allow_html=True)
        
        # Display original data
        orig_display = orig_df[selected_cols].iloc[start_row:end_row].copy()
        orig_display.insert(0, 'Row', range(start_row, min(end_row, len(orig_df))))
        
        st.dataframe(
            orig_display,
            use_container_width=True,
            height=600,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="data-panel modified-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📗 MODIFIED DATA</div>', unsafe_allow_html=True)
        
        # Display modified data with highlighting
        curr_display = curr_df[selected_cols].iloc[start_row:end_row].copy()
        curr_display.insert(0, 'Row', range(start_row, min(end_row, len(curr_df))))
        
        # Apply styling to highlight changes
        styled_curr = _style_changes(orig_df, curr_df, curr_display, selected_cols, start_row, end_row)
        
        st.dataframe(
            styled_curr,
            use_container_width=True,
            height=600,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
    <div style="margin-top: 20px; padding: 15px; background: #f9fafb; border-radius: 8px;">
        <strong>Legend:</strong>
        <span style="margin-left: 20px; padding: 4px 12px; background: #fef3c7; border-radius: 4px;">🟡 Modified Value</span>
        <span style="margin-left: 10px; padding: 4px 12px; background: #d1fae5; border-radius: 4px;">🟢 New Row</span>
        <span style="margin-left: 10px; padding: 4px 12px; background: #fee2e2; border-radius: 4px;">🔴 Removed Row</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Show change summary
    if comparison_data['changes_found']:
        st.divider()
        st.markdown("### 📊 Change Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Modified Cells", comparison_data['modified_cells'])
        with summary_col2:
            st.metric("Added Rows", comparison_data['added_rows'])
        with summary_col3:
            st.metric("Removed Rows", comparison_data['removed_rows'])


def _prepare_comparison_data(orig_df, curr_df, columns, start_row, end_row):
    """Prepare comparison data and detect changes"""
    
    changes_found = False
    modified_cells = 0
    added_rows = 0
    removed_rows = 0
    
    # Check for added/removed rows
    if end_row > len(orig_df):
        added_rows = min(end_row, len(curr_df)) - len(orig_df)
        changes_found = True
    
    if end_row > len(curr_df):
        removed_rows = min(end_row, len(orig_df)) - len(curr_df)
        changes_found = True
    
    # Check for modified cells
    for idx in range(start_row, min(end_row, len(orig_df), len(curr_df))):
        for col in columns:
            orig_val = orig_df.iloc[idx][col]
            curr_val = curr_df.iloc[idx][col]
            
            # Compare values (handle NaN)
            if pd.isna(orig_val) and pd.isna(curr_val):
                continue
            elif orig_val != curr_val:
                modified_cells += 1
                changes_found = True
    
    return {
        'changes_found': changes_found,
        'modified_cells': modified_cells,
        'added_rows': added_rows,
        'removed_rows': removed_rows
    }


def _style_changes(orig_df, curr_df, display_df, columns, start_row, end_row):
    """Apply styling to highlight changes in modified data"""
    
    def highlight_cell(row):
        styles = [''] * len(row)
        
        row_idx = row['Row']
        
        # Check if row is added (beyond original data)
        if row_idx >= len(orig_df):
            return ['background-color: #d1fae5'] * len(row)
        
        # Check if row is in current data
        if row_idx >= len(curr_df):
            return ['background-color: #fee2e2'] * len(row)
        
        # Compare each cell
        for i, col in enumerate(display_df.columns):
            if col == 'Row':
                continue
            
            if col in columns and row_idx < len(orig_df) and row_idx < len(curr_df):
                orig_val = orig_df.iloc[row_idx][col]
                curr_val = curr_df.iloc[row_idx][col]
                
                # Check if value changed
                if pd.isna(orig_val) and pd.isna(curr_val):
                    styles[i] = ''
                elif orig_val != curr_val:
                    styles[i] = 'background-color: #fef3c7; font-weight: bold'
        
        return styles
    
    return display_df.style.apply(highlight_cell, axis=1)
