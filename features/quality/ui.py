"""Data Quality Component - Enhanced with Top AI and Actions"""

import streamlit as st
import pandas as pd
import re
import io
from datetime import datetime

from state.session import show_toast


def render_data_quality():
    """Enhanced table with top AI assistant and actions"""
    
    state = st.session_state.app_state
    
    if state.df is None:
        st.info("📤 Please load data first")
        return
    
    # Custom CSS for faster animations
    st.markdown("""
        <style>
        div[data-testid="column"] {
            border: 1px solid #e0e0e0;
            padding: 8px;
            border-radius: 4px;
            background: #fafafa;
        }
        .stPopover {
            animation: fadeIn 0.1s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## ✨ Data Quality")
    
    # Initialize
    if 'reject_df' not in st.session_state:
        st.session_state.reject_df = pd.DataFrame()
    if 'validation_history' not in st.session_state:
        st.session_state.validation_history = []
    if 'dq_config' not in st.session_state:
        st.session_state.dq_config = {}
    
    # Top bar with metrics, AI, and actions
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Rows", len(state.df))
    
    with col2:
        st.metric("Columns", len(state.df.columns))
    
    with col3:
        st.metric("Rejected", len(st.session_state.reject_df))
    
    with col4:
        # AI Assistant at top
        with st.popover("🤖 AI Regex", use_container_width=True):
            st.markdown("**AI Regex Generator**")
            
            columns = state.df.columns.tolist()
            selected_col = st.selectbox("Column:", columns, key="ai_col")
            
            ai_q = st.text_area(
                "What to do?",
                placeholder="Remove special chars\nReplace _ with space\nExtract numbers",
                height=80,
                key="ai_q"
            )
            
            if st.button("✨ Generate", use_container_width=True, type="primary"):
                if selected_col and ai_q:
                    sug = _get_ai_suggestion(state.df, selected_col, ai_q)
                    if sug:
                        st.session_state.ai_sug = sug
                        st.rerun()
            
            if 'ai_sug' in st.session_state:
                sug = st.session_state.ai_sug
                st.success("✅ Generated")
                st.caption(sug['explanation'])
                
                if sug.get('pattern'):
                    st.code(sug['pattern'], language="regex")
                    st.text_input("Copy Pattern:", sug['pattern'], key="cp1")
                
                if sug.get('replace'):
                    st.code(sug['replace'])
                    st.text_input("Copy Replace:", sug['replace'], key="cp2")
                
                if st.button("❌ Clear", use_container_width=True):
                    del st.session_state.ai_sug
                    st.rerun()
    
    with col5:
        total_rules = sum(len(cfg.get('applied_rules', [])) for cfg in st.session_state.dq_config.values())
        if st.button(f"✅ Apply ({total_rules})", type="primary", use_container_width=True, disabled=total_rules == 0):
            _apply_all_rules(state)
            st.rerun()
    
    with col6:
        if st.session_state.validation_history:
            if st.button("↩️ Undo", use_container_width=True):
                _undo_last(state)
                st.rerun()
    
    st.divider()
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Enable All", use_container_width=True):
            for col in columns:
                st.session_state.dq_config[col]['enabled'] = True
            st.rerun()
    with col2:
        if st.button("❌ Disable All", use_container_width=True):
            for col in columns:
                st.session_state.dq_config[col]['enabled'] = False
            st.rerun()
    with col3:
        if st.button("🧹 Clear Rules", use_container_width=True):
            for col in columns:
                st.session_state.dq_config[col]['applied_rules'] = []
            st.rerun()
    with col4:
        if not st.session_state.reject_df.empty:
            if st.button("📥 Download", use_container_width=True):
                _download_rejected(st.session_state.reject_df)
    
    st.divider()
    
    # Table header (removed AI column, increased mode width)
    header_cols = st.columns([0.5, 1.2, 2, 1, 1.2, 1.8, 1, 0.8, 1])
    with header_cols[0]:
        st.markdown("**✓**")
    with header_cols[1]:
        st.markdown("**Column**")
    with header_cols[2]:
        st.markdown("**Values**")
    with header_cols[3]:
        st.markdown("**Rules**")
    with header_cols[4]:
        st.markdown("**Mode**")
    with header_cols[5]:
        st.markdown("**Configuration**")
    with header_cols[6]:
        st.markdown("**Preview**")
    with header_cols[7]:
        st.markdown("**Save**")
    with header_cols[8]:
        st.markdown("**Apply**")
    
    st.divider()
    
    # Table rows
    for idx, col_name in enumerate(columns):
        # Initialize config
        if col_name not in st.session_state.dq_config:
            st.session_state.dq_config[col_name] = {
                'enabled': False,
                'mode': 'Clean',
                'pattern': '',
                'replace': '',
                'case': 'UPPERCASE',
                'length_mode': 'Exact',
                'min_length': 0,
                'max_length': 50,
                'exact_length': 10,
                'applied_rules': []
            }
        
        config = st.session_state.dq_config[col_name]
        
        # Get sample values
        sample = state.df[col_name].dropna().astype(str).head(5).tolist()
        sample_str = ", ".join(sample[:5]) if sample else "No data"
        if len(sample_str) > 80:
            sample_str = sample_str[:80] + "..."
        
        # Create row
        row_cols = st.columns([0.5, 1.2, 2, 1, 1.2, 1.8, 1, 0.8, 1])
        
        # Column 0: Enable
        with row_cols[0]:
            enabled = st.checkbox("En", value=config['enabled'], key=f"en_{idx}", label_visibility="collapsed")
            config['enabled'] = enabled
        
        # Column 1: Column name
        with row_cols[1]:
            st.markdown(f"**{col_name}**")
        
        # Column 2: Values
        with row_cols[2]:
            st.caption(sample_str)
        
        # Column 3: Rules popover with edit/delete
        with row_cols[3]:
            rule_count = len(config['applied_rules'])
            if rule_count > 0:
                with st.popover(f"📋 {rule_count}", use_container_width=True):
                    st.markdown(f"**Rules ({rule_count})**")
                    for rule_idx, rule in enumerate(config['applied_rules']):
                        st.markdown(f"**{rule_idx + 1}. {rule['name']}**")
                        st.caption(f"Mode: {rule['mode']}")
                        if rule.get('pattern'):
                            st.caption(f"Pattern: {rule['pattern']}")
                        if rule.get('replace'):
                            st.caption(f"Replace: {rule['replace']}")
                        if rule['mode'] == 'Case':
                            st.caption(f"Case: {rule['case']}")
                        if rule['mode'] == 'Length':
                            st.caption(f"Length: {rule['length_mode']}")
                        st.caption(f"⏰ {rule['timestamp']}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("✏️ Edit", key=f"edit_{idx}_{rule_idx}", use_container_width=True):
                                # Load rule into config for editing
                                config['mode'] = rule['mode']
                                config['pattern'] = rule.get('pattern', '')
                                config['replace'] = rule.get('replace', '')
                                config['case'] = rule.get('case', 'UPPERCASE')
                                config['length_mode'] = rule.get('length_mode', 'Exact')
                                config['min_length'] = rule.get('min_length', 0)
                                config['max_length'] = rule.get('max_length', 50)
                                config['exact_length'] = rule.get('exact_length', 10)
                                # Remove from list
                                config['applied_rules'].pop(rule_idx)
                                show_toast(f"Editing: {rule['name']}", "info")
                                st.rerun()
                        with col_b:
                            if st.button("🗑️", key=f"del_{idx}_{rule_idx}", use_container_width=True):
                                config['applied_rules'].pop(rule_idx)
                                show_toast(f"Deleted: {rule['name']}", "info")
                                st.rerun()
                        
                        if rule_idx < len(config['applied_rules']) - 1:
                            st.divider()
            else:
                st.caption("No rules")
        
        # Column 4: Mode (increased width)
        with row_cols[4]:
            mode = st.selectbox(
                "Mode",
                ["Clean", "Replace", "Extract", "Validate", "Case", "Length"],
                index=["Clean", "Replace", "Extract", "Validate", "Case", "Length"].index(config['mode']),
                key=f"mode_{idx}",
                label_visibility="collapsed",
                disabled=not enabled
            )
            if mode != config['mode']:
                config['mode'] = mode
                st.rerun()
        
        # Column 5: Configuration
        with row_cols[5]:
            if mode in ["Clean", "Replace", "Extract", "Validate"]:
                pattern = st.text_input("Pattern", value=config['pattern'], placeholder="Regex", 
                                       key=f"pat_{idx}", label_visibility="collapsed", disabled=not enabled)
                config['pattern'] = pattern
                
                if mode == "Replace":
                    replace = st.text_input("Replace", value=config['replace'], placeholder="Text",
                                           key=f"rep_{idx}", label_visibility="collapsed", disabled=not enabled)
                    config['replace'] = replace
            
            elif mode == "Length":
                length_mode = st.selectbox("Length", ["Exact", "Minimum", "Maximum", "Range"],
                                          index=["Exact", "Minimum", "Maximum", "Range"].index(config['length_mode']),
                                          key=f"lm_{idx}", label_visibility="collapsed", disabled=not enabled)
                config['length_mode'] = length_mode
                
                if length_mode == "Exact":
                    exact = st.number_input("Exact", min_value=1, value=config['exact_length'],
                                           key=f"ex_{idx}", label_visibility="collapsed", disabled=not enabled)
                    config['exact_length'] = exact
                elif length_mode == "Minimum":
                    min_len = st.number_input("Min", min_value=0, value=config['min_length'],
                                             key=f"min_{idx}", label_visibility="collapsed", disabled=not enabled)
                    config['min_length'] = min_len
                elif length_mode == "Maximum":
                    max_len = st.number_input("Max", min_value=1, value=config['max_length'],
                                             key=f"max_{idx}", label_visibility="collapsed", disabled=not enabled)
                    config['max_length'] = max_len
                elif length_mode == "Range":
                    col_a, col_b = st.columns(2)
                    with col_a:
                        min_len = st.number_input("Min", min_value=0, value=config['min_length'],
                                                 key=f"minr_{idx}", label_visibility="collapsed", disabled=not enabled)
                        config['min_length'] = min_len
                    with col_b:
                        max_len = st.number_input("Max", min_value=1, value=config['max_length'],
                                                 key=f"maxr_{idx}", label_visibility="collapsed", disabled=not enabled)
                        config['max_length'] = max_len
            
            elif mode == "Case":
                case = st.selectbox("Case", ["UPPERCASE", "lowercase", "Title Case"],
                                   index=["UPPERCASE", "lowercase", "Title Case"].index(config['case']) if config['case'] in ["UPPERCASE", "lowercase", "Title Case"] else 0,
                                   key=f"case_{idx}", label_visibility="collapsed", disabled=not enabled)
                config['case'] = case
        
        # Column 6: Preview (faster animation)
        with row_cols[6]:
            with st.popover("👁️", disabled=not enabled, use_container_width=True):
                st.markdown("**Preview**")
                preview_df = _get_preview_dataframe(state.df, col_name, config)
                if preview_df is not None and not preview_df.empty:
                    st.dataframe(preview_df, use_container_width=True, height=250, hide_index=True)
                else:
                    st.caption("Configure first")
        
        # Column 7: Save
        with row_cols[7]:
            if st.button("💾", key=f"save_{idx}", disabled=not enabled, use_container_width=True):
                rule_name = _generate_rule_name(config)
                rule = {
                    'name': rule_name,
                    'mode': config['mode'],
                    'pattern': config['pattern'],
                    'replace': config['replace'],
                    'case': config['case'],
                    'length_mode': config['length_mode'],
                    'min_length': config['min_length'],
                    'max_length': config['max_length'],
                    'exact_length': config['exact_length'],
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                config['applied_rules'].append(rule)
                show_toast(f"Saved: {rule_name}", "success")
                st.rerun()
        
        # Column 8: Apply
        with row_cols[8]:
            if st.button("▶️", key=f"apply_{idx}", disabled=not enabled or not config['applied_rules'],
                        use_container_width=True):
                _apply_column_rules(state, col_name)
                st.rerun()
        
        st.divider()
    
    # History
    if st.session_state.validation_history:
        with st.expander(f"📜 History ({len(st.session_state.validation_history)})"):
            for idx, h in enumerate(reversed(st.session_state.validation_history)):
                st.caption(f"{len(st.session_state.validation_history)-idx}. {h['description']} - {h['timestamp']}")
    
    # Rejected
    if not st.session_state.reject_df.empty:
        with st.expander(f"🗑️ Rejected ({len(st.session_state.reject_df)})"):
            st.dataframe(st.session_state.reject_df.head(50), use_container_width=True)


def _generate_rule_name(config):
    """Auto-generate rule name"""
    mode = config['mode']
    
    if mode == "Clean":
        pattern = config.get('pattern', '')
        if pattern == '[^a-zA-Z0-9\\s]':
            return "Remove Special Chars"
        elif pattern == '\\s':
            return "Remove Spaces"
        elif pattern == '[0-9]':
            return "Remove Digits"
        else:
            return f"Clean: {pattern[:20]}"
    
    elif mode == "Replace":
        pattern = config.get('pattern', '')
        replace = config.get('replace', '')
        if pattern == '_' and replace == ' ':
            return "Replace _ with Space"
        else:
            return f"Replace: {pattern[:10]} → {replace[:10]}"
    
    elif mode == "Extract":
        pattern = config.get('pattern', '')
        if pattern == '[0-9]':
            return "Extract Digits"
        elif pattern == '[a-zA-Z]':
            return "Extract Letters"
        else:
            return f"Extract: {pattern[:20]}"
    
    elif mode == "Validate":
        pattern = config.get('pattern', '')
        if '@' in pattern:
            return "Validate Email"
        elif '\\d{10}' in pattern:
            return "Validate Phone"
        else:
            return "Validate Pattern"
    
    elif mode == "Case":
        case = config.get('case', 'UPPERCASE')
        return f"To {case}"
    
    elif mode == "Length":
        length_mode = config.get('length_mode', 'Exact')
        if length_mode == "Exact":
            return f"Length = {config.get('exact_length', 10)}"
        elif length_mode == "Minimum":
            return f"Length ≥ {config.get('min_length', 0)}"
        elif length_mode == "Maximum":
            return f"Length ≤ {config.get('max_length', 50)}"
        elif length_mode == "Range":
            return f"Length {config.get('min_length', 0)}-{config.get('max_length', 50)}"
    
    return f"{mode} Rule"


def _get_preview_dataframe(df, column, config):
    """Get preview dataframe"""
    try:
        mode = config['mode']
        pattern = config.get('pattern', '')
        replace = config.get('replace', '')
        case = config.get('case', 'UPPERCASE')
        
        sample = df[column].dropna().astype(str).head(10)
        if sample.empty:
            return None
        
        preview_data = []
        
        for val in sample:
            result = val
            status = "Valid"
            
            if mode == "Clean":
                if pattern:
                    result = re.sub(pattern, '', val)
            elif mode == "Replace":
                if pattern:
                    result = re.sub(pattern, replace, val)
            elif mode == "Extract":
                if pattern:
                    matches = re.findall(pattern, val)
                    result = ''.join(matches) if matches else "[REJECT]"
                    if result == "[REJECT]":
                        status = "Rejected"
            elif mode == "Validate":
                if pattern:
                    result = val if re.match(pattern, val) else "[REJECT]"
                    if result == "[REJECT]":
                        status = "Rejected"
            elif mode == "Case":
                if case == "UPPERCASE":
                    result = val.upper()
                elif case == "lowercase":
                    result = val.lower()
                elif case == "Title Case":
                    result = val.title()
            elif mode == "Length":
                length_mode = config['length_mode']
                val_len = len(val)
                
                if length_mode == "Exact":
                    if val_len != config['exact_length']:
                        result = "[REJECT]"
                        status = "Rejected"
                elif length_mode == "Minimum":
                    if val_len < config['min_length']:
                        result = "[REJECT]"
                        status = "Rejected"
                elif length_mode == "Maximum":
                    if val_len > config['max_length']:
                        result = "[REJECT]"
                        status = "Rejected"
                elif length_mode == "Range":
                    if val_len < config['min_length'] or val_len > config['max_length']:
                        result = "[REJECT]"
                        status = "Rejected"
            
            preview_data.append({
                'Before': val,
                'After': result,
                'Status': status
            })
        
        return pd.DataFrame(preview_data)
    
    except Exception as e:
        return None


def _get_ai_suggestion(df, column, user_question=""):
    """Get AI suggestion"""
    
    sample = df[column].dropna().astype(str).head(20).tolist()
    if not sample:
        return None
    
    has_special = any(re.search(r'[^a-zA-Z0-9\s]', v) for v in sample)
    has_underscore = any('_' in v for v in sample)
    has_email = any('@' in v for v in sample)
    has_digits = any(re.search(r'\d', v) for v in sample)
    
    if user_question:
        q = user_question.lower()
        
        if 'special' in q and 'remove' in q:
            return {
                'mode': 'Clean',
                'pattern': '[^a-zA-Z0-9\\s]',
                'explanation': 'Remove special characters'
            }
        elif 'underscore' in q and 'space' in q:
            return {
                'mode': 'Replace',
                'pattern': '_',
                'replace': ' ',
                'explanation': 'Replace underscores with spaces'
            }
        elif 'uppercase' in q or 'upper' in q:
            return {
                'mode': 'Case',
                'case': 'UPPERCASE',
                'explanation': 'Convert to UPPERCASE'
            }
        elif 'lowercase' in q or 'lower' in q:
            return {
                'mode': 'Case',
                'case': 'lowercase',
                'explanation': 'Convert to lowercase'
            }
        elif 'title' in q:
            return {
                'mode': 'Case',
                'case': 'Title Case',
                'explanation': 'Convert to Title Case'
            }
        elif 'number' in q or 'digit' in q:
            if 'extract' in q or 'only' in q:
                return {
                    'mode': 'Extract',
                    'pattern': '[0-9]',
                    'explanation': 'Extract only digits'
                }
        elif 'email' in q:
            return {
                'mode': 'Validate',
                'pattern': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$',
                'explanation': 'Validate email format'
            }
    
    # Auto-detect
    if has_underscore:
        return {
            'mode': 'Replace',
            'pattern': '_',
            'replace': ' ',
            'explanation': 'Replace underscores with spaces'
        }
    elif has_special:
        return {
            'mode': 'Clean',
            'pattern': '[^a-zA-Z0-9\\s]',
            'explanation': 'Remove special characters'
        }
    
    return {
        'mode': 'Clean',
        'pattern': '[^a-zA-Z0-9\\s]',
        'explanation': 'Remove special characters'
    }


def _apply_column_rules(state, column):
    """Apply rules for a single column"""
    config = st.session_state.dq_config[column]
    rules = config.get('applied_rules', [])
    
    if not rules:
        return
    
    backup_df = state.df.copy()
    backup_reject = st.session_state.reject_df.copy()
    
    rejected_rows = []
    
    for rule in rules:
        try:
            col_data = state.df[column].astype(str)
            mode = rule['mode']
            
            if mode == "Clean":
                state.df[column] = col_data.str.replace(rule['pattern'], '', regex=True)
            
            elif mode == "Replace":
                state.df[column] = col_data.str.replace(rule['pattern'], rule['replace'], regex=True)
            
            elif mode == "Extract":
                extracted = col_data.str.findall(rule['pattern'])
                state.df[column] = extracted.apply(lambda x: ''.join(x) if isinstance(x, list) else '')
                
                invalid_mask = state.df[column] == ''
                if invalid_mask.any():
                    rejected = state.df[invalid_mask].copy()
                    rejected['Rejection_Reason'] = f"{rule['name']} - No matches"
                    rejected['Rejected_Column'] = column
                    rejected['Rejected_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    rejected_rows.append(rejected)
                    state.df = state.df[~invalid_mask].reset_index(drop=True)
            
            elif mode == "Validate":
                invalid_mask = ~col_data.str.match(rule['pattern'], na=False)
                if invalid_mask.any():
                    rejected = state.df[invalid_mask].copy()
                    rejected['Rejection_Reason'] = f"{rule['name']} - Does not match"
                    rejected['Rejected_Column'] = column
                    rejected['Rejected_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    rejected_rows.append(rejected)
                    state.df = state.df[~invalid_mask].reset_index(drop=True)
            
            elif mode == "Case":
                if rule['case'] == "UPPERCASE":
                    state.df[column] = col_data.str.upper()
                elif rule['case'] == "lowercase":
                    state.df[column] = col_data.str.lower()
                elif rule['case'] == "Title Case":
                    state.df[column] = col_data.str.title()
            
            elif mode == "Length":
                lengths = col_data.str.len()
                length_mode = rule['length_mode']
                
                if length_mode == "Exact":
                    invalid_mask = lengths != rule['exact_length']
                elif length_mode == "Minimum":
                    invalid_mask = lengths < rule['min_length']
                elif length_mode == "Maximum":
                    invalid_mask = lengths > rule['max_length']
                elif length_mode == "Range":
                    invalid_mask = (lengths < rule['min_length']) | (lengths > rule['max_length'])
                
                if invalid_mask.any():
                    rejected = state.df[invalid_mask].copy()
                    rejected['Rejection_Reason'] = f"{rule['name']} - Length check failed"
                    rejected['Rejected_Column'] = column
                    rejected['Rejected_At'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    rejected_rows.append(rejected)
                    state.df = state.df[~invalid_mask].reset_index(drop=True)
        
        except Exception as e:
            show_toast(f"Error in rule '{rule['name']}': {str(e)}", "error")
            continue
    
    if rejected_rows:
        all_rejected = pd.concat(rejected_rows, ignore_index=True)
        if st.session_state.reject_df.empty:
            st.session_state.reject_df = all_rejected
        else:
            st.session_state.reject_df = pd.concat([st.session_state.reject_df, all_rejected], ignore_index=True)
    
    config['applied_rules'] = []
    
    st.session_state.validation_history.append({
        'description': f"Applied {len(rules)} rules to {column}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'rejected_count': len(all_rejected) if rejected_rows else 0,
        'backup_df': backup_df,
        'backup_reject_df': backup_reject
    })
    
    show_toast(f"Applied {len(rules)} rules to {column}", "success")


def _apply_all_rules(state):
    """Apply all rules"""
    enabled_cols = [col for col, cfg in st.session_state.dq_config.items() 
                    if cfg.get('enabled') and cfg.get('applied_rules')]
    
    if not enabled_cols:
        show_toast("No rules to apply", "warning")
        return
    
    for col in enabled_cols:
        _apply_column_rules(state, col)
    
    show_toast("Applied all rules", "success")


def _undo_last(state):
    """Undo last transformation"""
    if not st.session_state.validation_history:
        return
    
    last = st.session_state.validation_history.pop()
    state.df = last['backup_df'].copy()
    st.session_state.reject_df = last['backup_reject_df'].copy()
    
    show_toast("Undone", "info")


def _download_rejected(reject_df):
    """Download rejected records"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        reject_df.to_excel(writer, sheet_name='Rejected', index=False)
    
    output.seek(0)
    
    st.download_button(
        "⬇️ Download",
        data=output.getvalue(),
        file_name=f"rejected_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_rejected_btn"
    )
