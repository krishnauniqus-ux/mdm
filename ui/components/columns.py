"""Column management component"""

import streamlit as st
import pandas as pd
from core.transformations import (
    update_dataframe, 
    transform_handle_missing, 
    transform_convert_types,
    transform_standardize_text,
    transform_apply_regex,
    transform_apply_business_rule
)
from utils.data_utils import format_special_chars_display


def render_columns():
    """Column Management"""
    if st.session_state.df is None:
        st.info("Please upload data to manage columns")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📊 Column Management</div>', unsafe_allow_html=True)
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        filter_type = st.selectbox("Filter:", ["All", "With Issues", "Numeric", "Text", "High Missing %", "Low Cardinality"])
    with filter_col2:
        sort_by = st.selectbox("Sort by:", ["Name", "Missing %", "Uniqueness", "Memory"])
    with filter_col3:
        search = st.text_input("Search:", "")
    
    profiles = st.session_state.column_profiles
    
    if not profiles:
        st.warning("No column profiles available. Please refresh the data.")
        return
    
    # Apply filters
    filtered_profiles = profiles.copy()
    if filter_type == "With Issues":
        filtered_profiles = {k: v for k, v in profiles.items() if v.null_count > 0 or v.special_chars or v.outliers.get("count", 0) > 0}
    elif filter_type == "Numeric":
        filtered_profiles = {k: v for k, v in profiles.items() if 'int' in v.dtype or 'float' in v.dtype}
    elif filter_type == "Text":
        filtered_profiles = {k: v for k, v in profiles.items() if 'object' in v.dtype}
    elif filter_type == "High Missing %":
        filtered_profiles = {k: v for k, v in profiles.items() if v.null_percentage > 10}
    elif filter_type == "Low Cardinality":
        filtered_profiles = {k: v for k, v in profiles.items() if v.unique_percentage < 5}
    
    if search:
        filtered_profiles = {k: v for k, v in filtered_profiles.items() if search.lower() in k.lower()}
    
    # Sort
    if sort_by == "Missing %":
        filtered_profiles = dict(sorted(filtered_profiles.items(), key=lambda x: x[1].null_percentage, reverse=True))
    elif sort_by == "Uniqueness":
        filtered_profiles = dict(sorted(filtered_profiles.items(), key=lambda x: x[1].unique_percentage))
    elif sort_by == "Memory":
        def get_memory(profile):
            try:
                return float(profile.memory_usage.split()[0])
            except:
                return 0
        filtered_profiles = dict(sorted(filtered_profiles.items(), key=lambda x: get_memory(x[1])))
    
    # Bulk operations
    st.markdown("### Bulk Operations")
    bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
    
    with bulk_col1:
        selected_cols = st.multiselect("Select columns:", list(filtered_profiles.keys()), key="bulk_cols")
    
    with bulk_col2:
        bulk_action = st.selectbox("Action:", ["Drop Selected", "Fill Missing (Mean)", "Fill Missing (Median)", 
                                               "Convert to Category", "Standardize Text"])
    
    with bulk_col3:
        if st.button("Apply to Selected", type="primary", width="stretch"):
            if not selected_cols:
                st.warning("Please select columns first")
            elif bulk_action == "Drop Selected":
                df = st.session_state.df.drop(columns=selected_cols)
                update_dataframe(df, f"Dropped columns: {', '.join(selected_cols)}")
                st.rerun()
            elif "Fill Missing" in bulk_action:
                strategy = "mean" if "Mean" in bulk_action else "median"
                transform_handle_missing(strategy, selected_cols)
                st.rerun()
            elif bulk_action == "Convert to Category":
                conversions = {col: 'category' for col in selected_cols}
                transform_convert_types(conversions)
                st.rerun()
            elif bulk_action == "Standardize Text":
                transform_standardize_text(selected_cols, 'lower')
                st.rerun()
    
    st.divider()
    
    # Individual column details
    st.markdown(f"### Column Details ({len(filtered_profiles)} shown)")
    
    for col_name, profile in list(filtered_profiles.items())[:20]:  # Limit to 20
        with st.expander(f"{col_name} ({profile.dtype})"):
            
            # Basic Metrics Row
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Nulls", f"{profile.null_count} ({profile.null_percentage:.1f}%)")
            with metric_cols[1]:
                st.metric("Unique", f"{profile.unique_count} ({profile.unique_percentage:.1f}%)")
            with metric_cols[2]:
                st.metric("Memory", profile.memory_usage)
            with metric_cols[3]:
                if profile.outliers.get("count", 0) > 0:
                    st.metric("⚠️ Outliers", profile.outliers["count"])
                else:
                    st.metric("Outliers", "0")
            with metric_cols[4]:
                # Accuracy score
                acc_score = profile.accuracy_info.get('score', 100)
                color = "normal" if acc_score > 90 else "off" if acc_score > 70 else "error"
                st.metric("Accuracy", f"{acc_score:.0f}%", delta_color=color)
            
            # Data Quality Tabs
            detail_tabs = st.tabs(["📊 Overview", "🔍 Patterns", "✅ Quality", "🧹 Special Chars", "⚙️ Actions"])
            
            # Overview Tab
            with detail_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Data Type Information**")
                    st.write(f"• Type: `{profile.dtype}`")
                    st.write(f"• Total Rows: {profile.total_rows:,}")
                    st.write(f"• Duplicates: {profile.duplicate_count:,}")
                    
                    # Blank vs Null analysis
                    if profile.blank_count:
                        st.markdown("**Completeness Analysis**")
                        st.write(f"• Null values: {profile.blank_count.get('null_count', 0):,}")
                        st.write(f"• Empty strings: {profile.blank_count.get('empty_string_count', 0):,}")
                        st.write(f"• Whitespace only: {profile.blank_count.get('whitespace_only_count', 0):,}")
                        st.write(f"• **Total missing: {profile.blank_count.get('total_missing', 0):,}**")
                
                with col2:
                    st.markdown("**Sample Values**")
                    for i, val in enumerate(profile.sample_values[:5]):
                        st.write(f"{i+1}. `{val}`")
                    
                    # Formatting info
                    if profile.formatting_info:
                        st.markdown("**Formatting**")
                        fmt = profile.formatting_info
                        st.write(f"• Case consistency: {'✅ Yes' if fmt.get('consistent_case') else '❌ Mixed'}")
                        if fmt.get('case_type'):
                            st.write(f"• Dominant case: `{fmt['case_type']}`")
            
            # Patterns Tab
            with detail_tabs[1]:
                if profile.patterns:
                    patterns = profile.patterns
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Detected Patterns**")
                        if patterns.get('pattern_types'):
                            for ptype in patterns['pattern_types']:
                                st.write(f"• {ptype}")
                        else:
                            st.write("No specific patterns detected")
                        
                        # Pattern examples
                        if patterns.get('examples'):
                            st.markdown("**Examples**")
                            for ptype, example in patterns['examples'].items():
                                if example:
                                    st.write(f"• {ptype}: `{example}`")
                    
                    with col2:
                        # Length stats for strings
                        if 'length_stats' in patterns:
                            st.markdown("**Length Statistics**")
                            stats = patterns['length_stats']
                            st.write(f"• Min: {stats['min']} chars")
                            st.write(f"• Max: {stats['max']} chars")
                            st.write(f"• Avg: {stats['avg']} chars")
                        
                        # Range for numeric
                        if 'range' in patterns:
                            st.markdown("**Value Range**")
                            range_info = patterns['range']
                            st.write(f"• Min: {range_info['min']}")
                            st.write(f"• Max: {range_info['max']}")
                        
                        # Negative/Zero counts
                        if 'negative_count' in patterns:
                            st.write(f"• Negative values: {patterns['negative_count']:,}")
                            st.write(f"• Zero values: {patterns['zero_count']:,}")
                else:
                    st.info("No pattern analysis available for this column type")
            
            # Quality Tab
            with detail_tabs[2]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Accuracy Check**")
                    acc = profile.accuracy_info
                    score = acc.get('score', 100)
                    st.progress(score / 100)
                    st.write(f"Score: **{score:.1f}%**")
                    
                    if acc.get('issues'):
                        st.markdown("**Issues Found:**")
                        for issue in acc['issues']:
                            st.write(f"⚠️ {issue}")
                    else:
                        st.success("No accuracy issues detected")
                
                with col2:
                    st.markdown("**Validity Check**")
                    val = profile.validity_info
                    score = val.get('score', 100)
                    st.progress(score / 100)
                    st.write(f"Score: **{score:.1f}%**")
                    
                    if val.get('invalid_count', 0) > 0:
                        st.write(f"❌ Invalid count: {val['invalid_count']:,}")
                        if val.get('invalid_examples'):
                            for ex in val['invalid_examples']:
                                st.write(f"  • {ex}")
                    else:
                        st.success("All values are valid")
            
            # Special Characters Tab
            with detail_tabs[3]:
                if profile.special_chars:
                    st.markdown("**Special Characters Detected**")
                    
                    # Create a detailed table
                    char_data = []
                    for char_info in profile.special_chars:
                        char_data.append({
                            'Character': char_info['name'],
                            'Count': char_info['count'],
                            'Examples': char_info['examples'][:100] + '...' if len(char_info['examples']) > 100 else char_info['examples']
                        })
                    
                    if char_data:
                        st.dataframe(pd.DataFrame(char_data), width="stretch", hide_index=True)
                    
                    # Visualization of special chars
                    st.markdown("**Visual Representation**")
                    for char_info in profile.special_chars[:5]:
                        with st.container():
                            st.markdown(f"""
                            <div style="padding: 10px; border: 1px solid #e2e8f0; border-radius: 8px; margin: 5px 0;">
                                <b>{char_info['name']}</b> - Found {char_info['count']} times<br>
                                <small style="color: #64748b;">{char_info['examples'][:80]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.success("✅ No special characters found in this column")
            
            # Actions Tab - Regex & Business Rules
            with detail_tabs[4]:
                action_subtabs = st.tabs(["📝 Regex", "📋 Business Rules"])
                
                # Regex Subtab
                with action_subtabs[0]:
                    st.markdown("**Apply Regular Expression**")
                    
                    regex_col1, regex_col2 = st.columns(2)
                    with regex_col1:
                        regex_pattern = st.text_input("Pattern:", key=f"regex_pat_{col_name}", 
                                                     placeholder=r"e.g., ^[A-Z]{2}-\d{4}$")
                    with regex_col2:
                        regex_action = st.selectbox("Action:", 
                                                   ["replace", "extract", "match", "filter"],
                                                   key=f"regex_act_{col_name}")
                    
                    regex_replacement = ""
                    if regex_action == 'replace':
                        regex_replacement = st.text_input("Replacement:", "", key=f"regex_rep_{col_name}")
                    
                    # Test regex
                    if regex_pattern and st.button("Test Regex", key=f"regex_test_{col_name}"):
                        try:
                            sample_data = st.session_state.df[col_name].dropna().astype(str).head(10)
                            matches = sample_data.str.match(regex_pattern, na=False)
                            st.write(f"Matched {matches.sum()}/{len(sample_data)} sample rows")
                            st.write("Matching samples:", sample_data[matches].tolist())
                        except Exception as e:
                            st.error(f"Invalid regex: {e}")
                    
                    if st.button("Apply Regex", type="primary", key=f"regex_apply_{col_name}"):
                        if regex_pattern:
                            transform_apply_regex(col_name, regex_pattern, regex_replacement, regex_action)
                            st.success("✅ Regex applied!")
                            st.rerun()
                
                # Business Rules Subtab
                with action_subtabs[1]:
                    st.markdown("**Apply Business Rules**")
                    
                    rule_type = st.selectbox("Rule Type:", 
                                           ["range", "length", "email_format", "phone_format", 
                                            "date_range", "allowed_values", "pattern", "not_null", "unique"],
                                           key=f"rule_type_{col_name}")
                    
                    # Dynamic fields based on rule type
                    rule_kwargs = {}
                    
                    if rule_type == 'range':
                        rule_col1, rule_col2 = st.columns(2)
                        with rule_col1:
                            rule_kwargs['min_val'] = st.number_input("Min Value:", value=None, key=f"rule_min_{col_name}")
                        with rule_col2:
                            rule_kwargs['max_val'] = st.number_input("Max Value:", value=None, key=f"rule_max_{col_name}")
                    
                    elif rule_type == 'length':
                        rule_col1, rule_col2 = st.columns(2)
                        with rule_col1:
                            rule_kwargs['min_len'] = st.number_input("Min Length:", min_value=0, value=0, key=f"rule_minlen_{col_name}")
                        with rule_col2:
                            rule_kwargs['max_len'] = st.number_input("Max Length:", min_value=1, value=100, key=f"rule_maxlen_{col_name}")
                    
                    elif rule_type == 'date_range':
                        rule_col1, rule_col2 = st.columns(2)
                        with rule_col1:
                            rule_kwargs['min_date'] = st.date_input("Min Date:", key=f"rule_mindate_{col_name}")
                        with rule_col2:
                            rule_kwargs['max_date'] = st.date_input("Max Date:", key=f"rule_maxdate_{col_name}")
                    
                    elif rule_type == 'allowed_values':
                        allowed_input = st.text_area("Enter allowed values (one per line):", 
                                                    key=f"rule_allowed_{col_name}")
                        if allowed_input:
                            rule_kwargs['allowed_values'] = [v.strip() for v in allowed_input.split('\n') if v.strip()]
                    
                    elif rule_type == 'pattern':
                        rule_kwargs['pattern'] = st.text_input("Pattern:", key=f"rule_pattern_{col_name}")
                    
                    # Preview rule
                    if st.button("Preview Rule Impact", key=f"rule_preview_{col_name}"):
                        try:
                            df = st.session_state.df
                            original_count = len(df)
                            
                            # Simulate the rule
                            if rule_type == 'range' and pd.api.types.is_numeric_dtype(df[col_name]):
                                mask = pd.Series([True] * len(df))
                                if rule_kwargs.get('min_val') is not None:
                                    mask &= df[col_name] >= rule_kwargs['min_val']
                                if rule_kwargs.get('max_val') is not None:
                                    mask &= df[col_name] <= rule_kwargs['max_val']
                                remaining = mask.sum()
                            
                            elif rule_type == 'length':
                                lengths = df[col_name].astype(str).str.len()
                                mask = lengths.between(rule_kwargs.get('min_len', 0), 
                                                      rule_kwargs.get('max_len', float('inf')))
                                remaining = mask.sum()
                            
                            elif rule_type == 'not_null':
                                remaining = df[col_name].notna().sum()
                            
                            else:
                                remaining = "Preview not available for this rule type"
                            
                            if isinstance(remaining, int):
                                removed = original_count - remaining
                                st.info(f"Would keep {remaining:,} rows, remove {removed:,} rows ({removed/original_count*100:.1f}%)")
                            else:
                                st.info(remaining)
                                
                        except Exception as e:
                            st.error(f"Preview error: {e}")
                    
                    if st.button("Apply Business Rule", type="primary", key=f"rule_apply_{col_name}"):
                        transform_apply_business_rule(col_name, rule_type, **rule_kwargs)
                        st.success("✅ Business rule applied!")
                        st.rerun()
            
            # Quick Actions Row
            st.divider()
            action_cols = st.columns(4)
            with action_cols[0]:
                if st.button("🗑️ Drop Column", key=f"drop_{col_name}"):
                    df = st.session_state.df.drop(columns=[col_name])
                    update_dataframe(df, f"Dropped column {col_name}")
                    st.rerun()
            with action_cols[1]:
                new_type = st.selectbox("Convert", ["", "int", "float", "str", "datetime", "category", "bool"], 
                                       key=f"conv_{col_name}", label_visibility="collapsed")
                if new_type and st.button("Apply", key=f"apply_conv_{col_name}"):
                    transform_convert_types({col_name: new_type})
                    st.rerun()
            with action_cols[2]:
                if st.button("🧹 Clean Special", key=f"clean_spec_{col_name}"):
                    from core.transformations import transform_clean_special_chars
                    transform_clean_special_chars([col_name])
                    st.rerun()
            with action_cols[3]:
                if st.button("✂️ Trim", key=f"trim_{col_name}"):
                    from core.transformations import transform_trim_whitespace
                    transform_trim_whitespace([col_name])
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)