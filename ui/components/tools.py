"""Tools component"""

import streamlit as st
import numpy as np
from core.transformations import (
    transform_standardize_columns,
    transform_clean_special_chars,
    transform_trim_whitespace,
    transform_remove_exact_duplicates,
    transform_standardize_text,
    transform_handle_missing,
    transform_remove_outliers,
    transform_convert_types,
    transform_auto_fix,
    transform_apply_regex,
    transform_apply_business_rule
)
from utils.text_processing import AdvancedTitleCase


def render_tools():
    """Comprehensive Tools"""
    if st.session_state.df is None:
        st.info("Please upload data to use tools")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">🛠️ Data Standardization Tools</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🏷️ Column Names", "🧹 Clean Data", "📝 Text Case", "🔧 Handle Values", "⚡ Auto Fix", "🎯 Advanced Rules"])
    
    # TAB 0: Column Names
    with tabs[0]:
        st.write("### Standardize Column Names")
        convention = st.selectbox("Naming Convention:", 
                                 ["snake_case", "camelCase", "PascalCase", "lowercase", "UPPERCASE", "kebab-case"])
        
        preview_cols = list(st.session_state.df.columns[:3])
        st.info(f"Preview: {preview_cols}")
        
        if st.button("Apply Naming Convention", type="primary", width="stretch"):
            transform_standardize_columns(convention)
            st.success(f"✅ Converted to {convention}")
            st.rerun()
    
    # TAB 1: Clean Data
    with tabs[1]:
        st.write("### Clean Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Remove Special Characters", width="stretch"):
                transform_clean_special_chars()
                st.success("✅ Special characters cleaned")
                st.rerun()
        with col2:
            if st.button("✂️ Trim Whitespace", width="stretch"):
                transform_trim_whitespace()
                st.success("✅ Whitespace trimmed")
                st.rerun()
        
        st.divider()
        st.write("### Remove Duplicates")
        dup_type = st.radio("Type:", ["Exact", "Fuzzy"], horizontal=True)
        if dup_type == "Exact":
            if st.button("🗑️ Remove All Exact Duplicates", type="primary"):
                transform_remove_exact_duplicates()
                st.success("✅ Duplicates removed")
                st.rerun()
    
    # TAB 2: Text Case
    with tabs[2]:
        st.write("### Advanced Text Case")
        text_cols = st.multiselect("Select text columns:", 
                                  st.session_state.df.select_dtypes(include=['object']).columns.tolist(),
                                  key="text_case_cols")
        case_type = st.selectbox("Case type:", 
                                ["lower", "UPPER", "Title Case (APA)", "Title Case (Chicago)", 
                                 "Title Case (NYT)", "Sentence case"])
        
        if text_cols and st.button("Apply Text Case", type="primary"):
            style = 'apa' if 'APA' in case_type else 'chicago' if 'Chicago' in case_type else 'nyt' if 'NYT' in case_type else 'sentence'
            case = 'title' if 'Title' in case_type or 'Sentence' in case_type else case_type.lower()
            transform_standardize_text(text_cols, case, style)
            st.success("✅ Text case standardized")
            st.rerun()
        
        # Preview
        st.divider()
        st.write("### Preview Title Case")
        sample = st.text_input("Enter sample text:", "the quick brown fox jumps over the lazy dog")
        preview_style = st.selectbox("Style:", ["apa", "chicago", "nyt", "sentence"])
        result = AdvancedTitleCase.convert(sample, preview_style)
        st.success(f"Result: **{result}**")
    
    # TAB 3: Handle Values
    with tabs[3]:
        st.write("### Handle Missing Values")
        missing_strategy = st.selectbox("Strategy:", 
                                       ["auto", "drop rows", "mean", "median", "mode", "constant (0/UNKNOWN)"])
        target_cols = st.multiselect("Apply to (empty = all):", 
                                    st.session_state.df.columns.tolist(),
                                    key="missing_cols")
        
        if st.button("Apply Missing Value Strategy", type="primary"):
            strategy = missing_strategy.split()[0]
            transform_handle_missing(strategy, target_cols or None)
            st.success("✅ Missing values handled")
            st.rerun()
        
        st.divider()
        st.write("### Remove Outliers")
        outlier_cols = st.multiselect("Numeric columns:", 
                                     st.session_state.df.select_dtypes(include=[np.number]).columns.tolist(),
                                     key="outlier_cols")
        outlier_method = st.radio("Method:", ["IQR", "Z-Score"], horizontal=True)
        
        if st.button("📉 Remove Outliers"):
            method = "iqr" if "IQR" in outlier_method else "zscore"
            removed = transform_remove_outliers(outlier_cols or None, method)
            st.success(f"✅ Removed {removed} outliers")
            st.rerun()
        
        st.divider()
        st.write("### Convert Data Types")
        conv_col = st.selectbox("Column:", st.session_state.df.columns.tolist(), key="conv_col")
        conv_type = st.selectbox("Convert to:", ["int", "float", "str", "datetime", "category", "bool"])
        
        if st.button("🔄 Convert Type"):
            transform_convert_types({conv_col: conv_type})
            st.success(f"✅ Converted {conv_col} to {conv_type}")
            st.rerun()
    
    # TAB 4: Auto Fix
    with tabs[4]:
        st.write("### 🤖 Intelligent Auto-Fix")
        st.markdown("""
        This will execute:
        1. Remove exact duplicates
        2. Auto-fill missing values (smart detection)
        3. Clean special characters
        4. Trim whitespace
        5. Standardize column names to snake_case
        """)
        
        if st.button("⚡ RUN COMPLETE AUTO-FIX", type="primary", width="stretch"):
            with st.spinner("Applying intelligent fixes..."):
                transform_auto_fix()
            st.success("✅ Auto-fix completed!")
            st.balloons()
            st.rerun()
    
    # TAB 5: Advanced Rules (NEW)
    with tabs[5]:
        st.write("### 🎯 Advanced Regex & Business Rules")
        
        rule_col1, rule_col2 = st.columns(2)
        
        with rule_col1:
            st.markdown("**🔍 Regex Operations**")
            regex_target_col = st.selectbox("Target Column:", 
                                           st.session_state.df.columns.tolist(),
                                           key="tool_regex_col")
            regex_pattern = st.text_input("Regex Pattern:", 
                                         placeholder="e.g., ^[A-Za-z]+$",
                                         key="tool_regex_pattern")
            regex_action = st.selectbox("Action:", 
                                       ["replace", "extract", "match", "filter"],
                                       key="tool_regex_action")
            regex_replace = st.text_input("Replacement (if replace):", "", key="tool_regex_replace")
            
            # Test regex first
            if regex_pattern and st.button("🔍 Test Regex", key="test_regex_btn"):
                try:
                    import re
                    sample_data = st.session_state.df[regex_target_col].dropna().astype(str).head(10)
                    if regex_action == "match":
                        matches = sample_data.str.match(regex_pattern, na=False)
                        st.write(f"✅ Matched {matches.sum()}/{len(sample_data)} rows")
                        st.write("Matching values:", sample_data[matches].tolist())
                    elif regex_action == "filter":
                        matches = sample_data.str.contains(regex_pattern, na=False, regex=True)
                        st.write(f"✅ Would keep {matches.sum()}/{len(sample_data)} rows")
                    else:
                        # Show transformation preview
                        transformed = sample_data.apply(lambda x: re.sub(regex_pattern, regex_replace, x) if pd.notna(x) else x)
                        preview_df = pd.DataFrame({
                            'Original': sample_data,
                            'Transformed': transformed
                        })
                        st.write("Preview:")
                        st.dataframe(preview_df, width="stretch")
                except Exception as e:
                    st.error(f"❌ Regex error: {e}")
            
            if st.button("Apply Regex", type="primary", width="stretch"):
                if regex_pattern:
                    transform_apply_regex(regex_target_col, regex_pattern, regex_replace, regex_action)
                    st.success("✅ Regex applied!")
                    st.rerun()
        
        with rule_col2:
            st.markdown("**📋 Business Rules**")
            rule_target_col = st.selectbox("Target Column:", 
                                          st.session_state.df.columns.tolist(),
                                          key="tool_rule_col")
            rule_type = st.selectbox("Rule Type:", 
                                    ["range", "length", "email_format", "phone_format",
                                     "date_range", "allowed_values", "pattern", "not_null", "unique"],
                                    key="tool_rule_type")
            
            # Dynamic inputs based on rule type
            rule_kwargs = {}
            
            if rule_type == 'range':
                rule_col_min, rule_col_max = st.columns(2)
                with rule_col_min:
                    rule_kwargs['min_val'] = st.number_input("Min:", value=None, key="tool_rule_min")
                with rule_col_max:
                    rule_kwargs['max_val'] = st.number_input("Max:", value=None, key="tool_rule_max")
            
            elif rule_type == 'length':
                rule_col_min, rule_col_max = st.columns(2)
                with rule_col_min:
                    rule_kwargs['min_len'] = st.number_input("Min Len:", min_value=0, value=0, key="tool_rule_minlen")
                with rule_col_max:
                    rule_kwargs['max_len'] = st.number_input("Max Len:", min_value=1, value=255, key="tool_rule_maxlen")
            
            elif rule_type == 'date_range':
                rule_col_min, rule_col_max = st.columns(2)
                with rule_col_min:
                    rule_kwargs['min_date'] = st.date_input("Min Date:", key="tool_rule_mindate")
                with rule_col_max:
                    rule_kwargs['max_date'] = st.date_input("Max Date:", key="tool_rule_maxdate")
            
            elif rule_type == 'allowed_values':
                allowed = st.text_area("Allowed values (comma-separated):", key="tool_rule_allowed")
                if allowed:
                    rule_kwargs['allowed_values'] = [v.strip() for v in allowed.split(',') if v.strip()]
            
            elif rule_type == 'pattern':
                rule_kwargs['pattern'] = st.text_input("Pattern:", key="tool_rule_pattern")
            
            # Preview rule impact
            if st.button("👁️ Preview Impact", key="preview_rule_btn"):
                try:
                    df = st.session_state.df
                    original_count = len(df)
                    
                    if rule_type == 'range' and pd.api.types.is_numeric_dtype(df[rule_target_col]):
                        mask = pd.Series([True] * len(df))
                        if rule_kwargs.get('min_val') is not None:
                            mask &= df[rule_target_col] >= rule_kwargs['min_val']
                        if rule_kwargs.get('max_val') is not None:
                            mask &= df[rule_target_col] <= rule_kwargs['max_val']
                        remaining = mask.sum()
                        st.info(f"📊 Would keep {remaining:,} of {original_count:,} rows ({remaining/original_count*100:.1f}%)")
                    
                    elif rule_type == 'length':
                        lengths = df[rule_target_col].astype(str).str.len()
                        mask = lengths.between(rule_kwargs.get('min_len', 0), 
                                              rule_kwargs.get('max_len', float('inf')))
                        remaining = mask.sum()
                        st.info(f"📊 Would keep {remaining:,} of {original_count:,} rows ({remaining/original_count*100:.1f}%)")
                    
                    elif rule_type == 'email_format':
                        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        mask = df[rule_target_col].astype(str).str.match(email_pattern, na=False)
                        remaining = mask.sum()
                        st.info(f"📊 {remaining:,} valid emails found ({remaining/original_count*100:.1f}%)")
                    
                    elif rule_type == 'not_null':
                        remaining = df[rule_target_col].notna().sum()
                        st.info(f"📊 {remaining:,} non-null values ({remaining/original_count*100:.1f}%)")
                    
                    elif rule_type == 'unique':
                        unique_count = df[rule_target_col].nunique()
                        st.info(f"📊 {unique_count:,} unique values out of {original_count:,}")
                    
                    else:
                        st.info("ℹ️ Preview not available for this rule type")
                        
                except Exception as e:
                    st.error(f"Preview error: {e}")
            
            if st.button("Apply Business Rule", type="primary", width="stretch"):
                transform_apply_business_rule(rule_target_col, rule_type, **rule_kwargs)
                st.success("✅ Business rule applied!")
                st.rerun()
        
        # Common regex patterns helper
        st.divider()
        st.markdown("### 📚 Common Regex Patterns Library")
        
        common_patterns = {
            "Email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "Phone (US)": r'^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$',
            "URL": r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            "IP Address": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            "ZIP Code": r'^\d{5}(-\d{4})?$',
            "Credit Card": r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
            "Date (MM/DD/YYYY)": r'^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}$',
            "Only Letters": r'^[A-Za-z]+$',
            "Only Numbers": r'^\d+$',
            "Alphanumeric": r'^[A-Za-z0-9]+$',
            "HTML Tags": r'<[^>]+>',
            "Whitespace": r'\s+'
        }
        
        pattern_cols = st.columns(3)
        for i, (name, pattern) in enumerate(common_patterns.items()):
            with pattern_cols[i % 3]:
                with st.expander(f"📋 {name}"):
                    st.code(pattern, language="regex")
                    if st.button(f"Use This Pattern", key=f"use_pat_{i}"):
                        st.session_state.tool_regex_pattern = pattern
                        st.toast(f"Pattern for {name} copied to Regex Pattern field!")
                        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)