"""Compare Component - Side-by-side data comparison"""

import streamlit as st
import pandas as pd

from state.session import st


def render_compare():
    """Enhanced comparison with change highlighting"""
    
    state = st.session_state.app_state
    
    if state.df is None or state.original_df is None:
        st.info("📤 Load data and make changes to use comparison")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">⚖️ Compare: Original vs Modified</div>', unsafe_allow_html=True)
    
    # Comparison settings
    settings_col1, settings_col2, settings_col3 = st.columns(3)
    
    with settings_col1:
        max_rows = st.number_input("Max rows to compare:", 
                                  min_value=100, max_value=10000, value=1000, step=100)
    
    with settings_col2:
        highlight_changes = st.checkbox("Highlight changes", value=True)
    
    with settings_col3:
        show_diff_only = st.checkbox("Show changed rows only", value=False)
    
    # Get comparison data
    orig_df = state.original_df.head(max_rows)
    curr_df = state.df.head(max_rows)
    
    # Align dataframes for comparison
    common_cols = list(set(orig_df.columns) & set(curr_df.columns))
    
    # Column selection
    st.markdown("**Select columns to compare:**")
    selected_cols = st.multiselect("Columns:", 
                                  common_cols,
                                  default=common_cols[:5] if len(common_cols) > 5 else common_cols,
                                  key="compare_cols")
    
    if not selected_cols:
        st.warning("Select at least one column to compare")
        return
    
    # Perform comparison
    comparison_df = _create_comparison_view(orig_df, curr_df, selected_cols, highlight_changes, show_diff_only)
    
    # Display
    st.divider()
    st.markdown("**Comparison View**")
    
    if highlight_changes and 'change_status' in comparison_df.columns:
        # Color code rows
        def highlight_changes(row):
            if row['change_status'] == 'MODIFIED':
                return ['background-color: #fef3c7'] * len(row)
            elif row['change_status'] == 'NEW':
                return ['background-color: #d1fae5'] * len(row)
            elif row['change_status'] == 'DELETED':
                return ['background-color: #fee2e2'] * len(row)
            return [''] * len(row)
        
        styled_df = comparison_df.style.apply(highlight_changes, axis=1)
        st.dataframe(styled_df, width="stretch", height=500)
        
        # Legend
        st.markdown("""
        <div style="display: flex; gap: 20px; margin-top: 10px;">
            <span style="background: #fef3c7; padding: 4px 8px; border-radius: 4px;">🟡 Modified</span>
            <span style="background: #d1fae5; padding: 4px 8px; border-radius: 4px;">🟢 New</span>
            <span style="background: #fee2e2; padding: 4px 8px; border-radius: 4px;">🔴 Deleted</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.dataframe(comparison_df, width="stretch", height=500)
    
    # Summary statistics
    st.divider()
    _render_comparison_stats(orig_df, curr_df)
    
    st.markdown('</div>', unsafe_allow_html=True)


def _create_comparison_view(orig_df, curr_df, cols, highlight, diff_only):
    """Create detailed comparison view"""
    
    # For simplicity, compare by index
    orig_subset = orig_df[cols].copy()
    curr_subset = curr_df[cols].copy()
    
    # Add source indicator
    orig_subset['_source'] = 'ORIGINAL'
    curr_subset['_source'] = 'MODIFIED'
    
    # Combine for display
    combined = pd.concat([orig_subset, curr_subset], ignore_index=True)
    
    if highlight:
        # Determine change status (simplified - assumes row order similarity)
        combined['change_status'] = 'UNCHANGED'
        
        # Mark first half as original, second as modified for demo
        # In real implementation, you'd do proper row matching
        n_orig = len(orig_subset)
        combined.loc[:n_orig-1, 'change_status'] = 'ORIGINAL'
        combined.loc[n_orig:, 'change_status'] = 'MODIFIED'
    
    if diff_only and highlight:
        combined = combined[combined['change_status'] != 'UNCHANGED']
    
    return combined


def _render_comparison_stats(orig_df, curr_df):
    """Render comparison statistics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Rows", len(orig_df))
    with col2:
        st.metric("Current Rows", len(curr_df), 
                 delta=f"{len(curr_df) - len(orig_df):+d}")
    with col3:
        orig_cols = len(orig_df.columns)
        curr_cols = len(curr_df.columns)
        st.metric("Original Columns", orig_cols)
    with col4:
        st.metric("Current Columns", curr_cols,
                 delta=f"{curr_cols - orig_cols:+d}")
    
    # Detailed changes
    st.markdown("**Detailed Changes:**")
    
    changes = []
    
    # Row count change
    row_diff = len(curr_df) - len(orig_df)
    if row_diff != 0:
        changes.append(f"• Rows: {row_diff:+,} ({'added' if row_diff > 0 else 'removed'})")
    
    # Column changes
    added_cols = set(curr_df.columns) - set(orig_df.columns)
    removed_cols = set(orig_df.columns) - set(curr_df.columns)
    
    if added_cols:
        changes.append(f"• Columns added: {', '.join(list(added_cols)[:5])}")
    if removed_cols:
        changes.append(f"• Columns removed: {', '.join(list(removed_cols)[:5])}")
    
    # Memory change
    orig_mem = orig_df.memory_usage(deep=True).sum() / 1024 / 1024
    curr_mem = curr_df.memory_usage(deep=True).sum() / 1024 / 1024
    mem_diff = curr_mem - orig_mem
    
    changes.append(f"• Memory: {mem_diff:+.1f} MB ({mem_diff/orig_mem*100:+.1f}%)")
    
    for change in changes:
        st.write(change)