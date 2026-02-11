"""Data Quality Component - Enterprise data quality rules with preview"""

import streamlit as st
import pandas as pd
import re

from state.session import st, update_dataframe, show_toast
from core.transformations import (
    transform_apply_regex,
    transform_apply_business_rule,
    transform_handle_missing,
    transform_standardize_text,
    transform_trim_whitespace,
    transform_clean_special_chars
)


def render_data_quality():
    """Enterprise Data Quality with live preview and business rules"""
    
    state = st.session_state.app_state
    
    if state.df is None:
        st.info("📤 Please load data first")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">✨ Data Quality & Business Rules</div>', unsafe_allow_html=True)
    
    # Compact workflow indicator
    st.markdown("""
    <div style="display: flex; justify-content: space-between; gap: 8px; margin-bottom: 12px; padding: 0;">
        <div style="text-align: center; flex: 1; padding: 6px 8px; background: #dbeafe; border-radius: 6px; font-size: 0.75rem;">
            <b>1. Select</b>
        </div>
        <div style="text-align: center; flex: 1; padding: 6px 8px; background: #f3f4f6; border-radius: 6px; font-size: 0.75rem;">
            <b>2. Configure</b>
        </div>
        <div style="text-align: center; flex: 1; padding: 6px 8px; background: #f3f4f6; border-radius: 6px; font-size: 0.75rem;">
            <b>3. Preview</b>
        </div>
        <div style="text-align: center; flex: 1; padding: 6px 8px; background: #f3f4f6; border-radius: 6px; font-size: 0.75rem;">
            <b>4. Apply</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main interface
    dq_tabs = st.tabs([
        "🧹 Quick Fixes",
        "📝 Regex Rules", 
        "📋 Business Rules",
        "⚡ Custom Expressions",
        "📊 Preview & Apply"
    ])
    
    with dq_tabs[0]:
        _render_quick_fixes()
    
    with dq_tabs[1]:
        _render_regex_rules()
    
    with dq_tabs[2]:
        _render_business_rules()
    
    with dq_tabs[3]:
        _render_custom_expressions()
    
    with dq_tabs[4]:
        _render_preview_apply()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_quick_fixes():
    """Common quick fix operations - compact layout"""
    state = st.session_state.app_state
    
    st.markdown("### 🧹 Quick Fixes")
    
    # Use two columns for better organization
    fix_col1, fix_col2, fix_col3 = st.columns(3)
    
    with fix_col1:
        if st.button("✂️ Trim Whitespace", width="stretch", key="trim_btn"):
            # Sync session state for transformation functions
            st.session_state.df = state.df.copy()
            transform_trim_whitespace()
            state.df = st.session_state.df.copy()
            show_toast("Whitespace trimmed from all columns", "success")
        
        if st.button("🧹 Clean Special Chars", width="stretch", key="clean_btn"):
            # Sync session state for transformation functions
            st.session_state.df = state.df.copy()
            transform_clean_special_chars()
            state.df = st.session_state.df.copy()
            show_toast("Special characters cleaned", "success")
    
    with fix_col2:
        missing_strategy = st.selectbox("Handle Missing:", 
                                       ["mean", "median", "mode", "constant"],
                                       key="quick_missing", label_visibility="collapsed")
        if st.button("Apply Strategy", width="stretch", key="missing_btn"):
            # Sync session state for transformation functions
            st.session_state.df = state.df.copy()
            transform_handle_missing(missing_strategy)
            state.df = st.session_state.df.copy()
            show_toast(f"Missing values filled with {missing_strategy}", "success")
    
    with fix_col3:
        text_action = st.selectbox("Text Case:", 
                                  ["Lowercase", "Uppercase", "Title Case", "Sentence Case"],
                                  key="quick_text_action", label_visibility="collapsed")
        if st.button("Standardize Text", width="stretch", key="text_btn"):
            case_map = {
                "Lowercase": "lower",
                "Uppercase": "upper", 
                "Title Case": "title",
                "Sentence Case": "sentence"
            }
            # Get text columns once
            text_cols = state.df.select_dtypes(include=['object']).columns.tolist()
            # Sync session state for transformation functions
            st.session_state.df = state.df.copy()
            transform_standardize_text(text_cols, case_map[text_action])
            state.df = st.session_state.df.copy()
            show_toast(f"Text standardized to {text_action}", "success")


def _render_regex_rules():
    """Regex-based transformations with live preview"""
    state = st.session_state.app_state
    
    st.markdown("### 📝 Regex Rules")
    
    # Compact column selection and regex input
    col1, col2 = st.columns([1, 2])
    
    with col1:
        target_col = st.selectbox("Column:", 
                                 state.df.columns.tolist(),
                                 key="regex_target_col", label_visibility="collapsed")
    
    with col2:
        regex_pattern = st.text_input("Pattern:", 
                                     placeholder="e.g., ^[A-Z]{2}-\d{4}$",
                                     key="regex_pattern_input",
                                     label_visibility="collapsed")
    
    # Sample data preview (compact)
    with st.expander("👁️ Sample Data", expanded=False):
        st.dataframe(state.df[[target_col]].head(10), use_container_width=True, height=200)
    
    # Regex action and preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        replacement = ""
        regex_action = st.selectbox("Action:", 
                                   ["replace", "extract", "match", "filter", "remove"],
                                   key="regex_action_select",
                                   label_visibility="collapsed")
        if regex_action == "replace":
            replacement = st.text_input("Replace With:", "", key="regex_replacement", label_visibility="collapsed")
    
    with col2:
        st.write("")  # Spacing
    
    # Live preview
    if regex_pattern:
        try:
            pattern = re.compile(regex_pattern)
            sample_df = state.df[[target_col]].head(50).copy()
            
            if regex_action == "replace":
                sample_df['Preview'] = sample_df[target_col].astype(str).str.replace(regex_pattern, replacement, regex=True)
            elif regex_action == "extract":
                extracted = sample_df[target_col].astype(str).str.extract(regex_pattern)
                sample_df['Preview'] = extracted[0] if not extracted.empty else None
            elif regex_action == "match":
                mask = sample_df[target_col].astype(str).str.match(regex_pattern, na=False)
                sample_df['Preview'] = sample_df[target_col].where(mask, '[REMOVED]')
            elif regex_action == "filter":
                mask = sample_df[target_col].astype(str).str.contains(regex_pattern, na=False, regex=True)
                sample_df['Preview'] = sample_df[target_col].where(~mask, '[REMOVED]')
            elif regex_action == "remove":
                sample_df['Preview'] = sample_df[target_col].astype(str).str.replace(regex_pattern, '', regex=True)
            
            st.dataframe(sample_df, use_container_width=True, height=250)
            
            changed = (sample_df[target_col] != sample_df['Preview']).sum()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📊 {changed} of {len(sample_df)} rows would change ({changed/len(sample_df)*100:.1f}%)")
            with col2:
                if st.button("✅ Apply", key="apply_regex", width="stretch"):
                    # Sync session state for transformation functions
                    st.session_state.df = state.df.copy()
                    transform_apply_regex(target_col, regex_pattern, replacement, regex_action)
                    state.df = st.session_state.df.copy()
                    st.rerun()
                
        except re.error as e:
            st.error(f"❌ Invalid regex: {e}")
    else:
        st.caption("Enter a regex pattern to see preview")
    
    # Common patterns (compact)
    st.caption("**Quick Patterns:** ", help="Click to auto-fill the pattern field")
    patterns = {
        "Email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "Phone": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        "URL": r'https?://\S+',
        "Date": r'\d{1,2}/\d{1,2}/\d{4}',
        "Multi-Space": r'\s{2,}',
    }
    
    pat_cols = st.columns(5)
    for i, (name, pat) in enumerate(patterns.items()):
        with pat_cols[i % 5]:
            if st.button(f"📋 {name}", key=f"pat_{i}", width="stretch"):
                st.session_state.regex_pattern_input = pat
                st.rerun()
    
def _render_business_rules():
    """Business rule engine - compact"""
    state = st.session_state.app_state
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        rule_col = st.selectbox("Column:", 
                               state.df.columns.tolist(),
                               key="biz_rule_col", label_visibility="collapsed")
    
    with col2:
        rule_type = st.selectbox("Rule Type:", 
                                ["range", "length", "email_format", "phone_format",
                                 "date_range", "allowed_values", "not_null", "unique", "pattern"],
                                key="biz_rule_type",
                                label_visibility="collapsed",
                                format_func=lambda x: {
                                    "range": "Numeric Range",
                                    "length": "String Length",
                                    "email_format": "Email Format",
                                    "phone_format": "Phone Format", 
                                    "date_range": "Date Range",
                                    "allowed_values": "Allowed Values",
                                    "not_null": "Not Null",
                                    "unique": "Unique Only",
                                    "pattern": "Pattern Match"
                                }.get(x, x))
    
    # Dynamic parameters (compact)
    rule_params = {}
    
    if rule_type == "range":
        col1, col2 = st.columns(2)
        with col1:
            rule_params['min_val'] = st.number_input("Min:", value=None, key="biz_min", label_visibility="collapsed")
        with col2:
            rule_params['max_val'] = st.number_input("Max:", value=None, key="biz_max", label_visibility="collapsed")
    
    elif rule_type == "length":
        col1, col2 = st.columns(2)
        with col1:
            rule_params['min_len'] = st.number_input("Min Len:", min_value=0, value=0, key="biz_minlen", label_visibility="collapsed")
        with col2:
            rule_params['max_len'] = st.number_input("Max Len:", min_value=1, value=255, key="biz_maxlen", label_visibility="collapsed")
    
    elif rule_type == "date_range":
        col1, col2 = st.columns(2)
        with col1:
            rule_params['min_date'] = st.date_input("From:", key="biz_mindate", label_visibility="collapsed")
        with col2:
            rule_params['max_date'] = st.date_input("To:", key="biz_maxdate", label_visibility="collapsed")
    
    elif rule_type == "allowed_values":
        values_input = st.text_area("Values (one per line):", 
                                   key="biz_allowed",
                                   height=80,
                                   label_visibility="collapsed")
        if values_input:
            rule_params['allowed_values'] = [v.strip() for v in values_input.split('\n') if v.strip()]
    
    elif rule_type == "pattern":
        rule_params['pattern'] = st.text_input("Regex:", key="biz_pattern", label_visibility="collapsed")
    
    # Action on violation (compact)
    col1, col2 = st.columns([2, 1])
    with col1:
        violation_action = st.selectbox("On Violation:", 
                                       ["remove_rows", "flag_column", "fill_default", "stop"],
                                       key="biz_violation",
                                       label_visibility="collapsed",
                                       format_func=lambda x: {
                                           "remove_rows": "Remove rows",
                                           "flag_column": "Flag column",
                                           "fill_default": "Fill default",
                                           "stop": "Validate only"
                                       }.get(x, x))
    
    if violation_action == "fill_default":
        with col2:
            rule_params['default_value'] = st.text_input("Default:", key="biz_default", label_visibility="collapsed")
    else:
        with col2:
            st.write("")
    
    # Actions (horizontal buttons)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👁️ Preview", width="stretch", key="preview_rule"):
            _preview_business_rule(rule_col, rule_type, rule_params)
    
    with col2:
        if st.button("✅ Apply Rule", type="primary", width="stretch", key="apply_rule"):
            state = st.session_state.app_state
            # Sync session state for transformation functions
            st.session_state.df = state.df.copy()
            transform_apply_business_rule(rule_col, rule_type, **rule_params)
            state.df = st.session_state.df.copy()
            st.rerun()


def _preview_business_rule(col, rule_type, params):
    """Preview business rule impact without applying"""
    state = st.session_state.app_state
    
    try:
        df = state.df
        original_count = len(df)
        
        if rule_type == 'range' and pd.api.types.is_numeric_dtype(df[col]):
            mask = pd.Series([True] * len(df))
            if params.get('min_val') is not None:
                mask &= df[col] >= params['min_val']
            if params.get('max_val') is not None:
                mask &= df[col] <= params['max_val']
            remaining = mask.sum()
            
        elif rule_type == 'length':
            lengths = df[col].astype(str).str.len()
            mask = lengths.between(params.get('min_len', 0), params.get('max_len', float('inf')))
            remaining = mask.sum()
            
        elif rule_type == 'email_format':
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            mask = df[col].astype(str).str.match(pattern, na=False)
            remaining = mask.sum()
            
        elif rule_type == 'not_null':
            mask = df[col].notna() & (df[col] != '')
            remaining = mask.sum()
            
        elif rule_type == 'unique':
            remaining = df[col].nunique()
            st.caption(f"📊 {remaining:,} unique | {len(df):,} total")
            return
        
        else:
            st.caption("Preview not available for this rule type")
            return
        
        removed = original_count - remaining
        pct_kept = (remaining / original_count * 100) if original_count > 0 else 0
        
        st.progress(pct_kept / 100)
        st.caption(f"✅ Keep: {remaining:,} ({pct_kept:.1f}%) | ❌ Remove: {removed:,} ({100-pct_kept:.1f}%)")
        
        # Show sample violations
        if removed > 0:
            violations = df[~mask][[col]].head(10)
            with st.expander("View sample violations"):
                st.dataframe(violations, width="stretch")
                
    except Exception as e:
        st.error(f"Preview error: {e}")


def _render_custom_expressions():
    """Pandas expression evaluator - compact"""
    state = st.session_state.app_state
    
    st.markdown("### ⚡ Custom Expressions")
    
    st.caption("Write Pandas expressions. Use `df` to reference the dataframe. E.g., `df['new'] = df['col1'] + df['col2']`")
    
    expression = st.text_area("Expression:", 
                             height=80,
                             placeholder="df['new_col'] = df['col1'] + df['col2']",
                             key="custom_expr",
                             label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👁️ Preview", width="stretch", key="preview_expr"):
            try:
                df_copy = state.df.head(100).copy()
                local_vars = {'df': df_copy, 'pd': pd, 'np': __import__('numpy')}
                exec(expression, {"__builtins__": {}}, local_vars)
                
                st.success("✅ Valid")
                st.dataframe(df_copy.head(10), use_container_width=True, height=200)
                
            except Exception as e:
                st.error(f"❌ {str(e)[:100]}")
    
    with col2:
        if st.button("✅ Apply", type="primary", width="stretch", key="apply_expr"):
            try:
                local_vars = {'df': state.df, 'pd': pd, 'np': __import__('numpy')}
                exec(expression, {"__builtins__": {}}, local_vars)
                # Sync session state for update_dataframe
                st.session_state.df = state.df.copy()
                st.session_state.fixes_applied = getattr(st.session_state, 'fixes_applied', [])
                st.session_state.last_operation = ""
                st.session_state.operation_count = getattr(st.session_state, 'operation_count', 0)
                update_dataframe(local_vars['df'], f"Custom expression applied")
                state.df = st.session_state.df.copy()
                st.rerun()
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)[:100]}")


def _render_preview_apply():
    """Final preview and apply changes - compact"""
    state = st.session_state.app_state
    
    st.markdown("### 📊 Changes & Statistics")
    
    if not state.fixes_applied:
        st.caption("No operations applied yet")
        return
    
    # Compact operation history
    st.markdown("**Operations:**")
    ops_list = []
    for op in reversed(state.fixes_applied[-5:]):
        ops_list.append(f"• {op['operation']}")
    st.caption("\n".join(ops_list))
    
    # Statistics comparison (compact)
    if state.original_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        orig_rows, curr_rows = len(state.original_df), len(state.df)
        orig_cols, curr_cols = len(state.original_df.columns), len(state.df.columns)
        
        with col1:
            st.metric("Rows", curr_rows, delta=f"{curr_rows-orig_rows:+d}")
        with col2:
            st.metric("Columns", curr_cols, delta=f"{curr_cols-orig_cols:+d}")
        with col3:
            orig_missing = state.original_df.isnull().sum().sum()
            curr_missing = state.df.isnull().sum().sum()
            st.metric("Missing", curr_missing, delta=f"{curr_missing-orig_missing:+d}")
        with col4:
            orig_mem = round(state.original_df.memory_usage(deep=True).sum() / 1024 / 1024, 1)
            curr_mem = round(state.df.memory_usage(deep=True).sum() / 1024 / 1024, 1)
            st.metric("Memory (MB)", curr_mem, delta=f"{curr_mem-orig_mem:+.1f}")
        
        # Side-by-side comparison
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("**Original** (first 10 rows)")
            st.dataframe(state.original_df.head(10), use_container_width=True, height=250)
        with comp_col2:
            st.markdown("**Current** (first 10 rows)")
            st.dataframe(state.df.head(10), use_container_width=True, height=250)
    
    # Reset option
    st.divider()
    if st.button("🔄 Reset All Changes", type="secondary", width="stretch"):
        state.df = state.original_df.copy()
        state.fixes_applied = []
        state.operation_count = 0
        show_toast("Changes reset to original", "success")
        st.rerun()