"""Data Profiling Component - Enhanced Version with Pattern Detection"""

import streamlit as st
import pandas as pd
import io
import re
import unicodedata
from datetime import datetime
from collections import Counter, defaultdict
import time
import numpy as np

# Import session utilities
from state.session import st, show_toast
from core.profiler import DataProfilerEngine

# ==========================================
# ENHANCED PATTERN DETECTION REGEX LIBRARY
# ==========================================

PATTERN_LIBRARY = {
    'Email': {
        'regex': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        'description': 'Email address (global)',
        'category': 'Contact'
    },
    'PAN (India)': {
        'regex': r'(?<!\w)[A-Z]{5}\d{4}[A-Z](?!\w)',
        'description': 'Permanent Account Number (India)',
        'category': 'Government ID'
    },
    'Aadhaar (India)': {
        'regex': r'(?<!\d)(?:\d[\s\-._–—()]){12}(?!\d)',
        'description': '12-digit Aadhaar ID with any spacing',
        'category': 'Government ID',
        'variations': [
            '123456789874',
            '1234 5678 9874',
            '1234-5678-9874',
            '1234_5678_9874',
            '12 34 56 78 98 74',
            '1 2 3 4 5 6 7 8 9 8 7 4',
            '1234.5678.9874'
        ]
    },
    'Passport (Global)': {
        'regex': r'(?<!\w)[A-Z0-9]{6,9}(?!\w)',
        'description': 'Passport number (generic)',
        'category': 'Government ID'
    },
    'Credit/Debit Card': {
        'regex': r'(?<!\d)(?:\d[\s\-._]){13,19}(?!\d)',
        'description': '13-19 digit card number',
        'category': 'Financial',
        'variations': [
            '4111111111111111',
            '4111 1111 1111 1111',
            '4111-1111-1111-1111'
        ]
    },
    'Phone - Mobile (Global)': {
        'regex': r'(?<!\d)(?:\+?\d[\s\-()]){10,15}(?!\d)',
        'description': 'Mobile numbers with country code',
        'category': 'Contact',
        'variations': [
            '+91 98765 43210',
            '919876543210',
            '+1-234-567-8900'
        ]
    },
    'Phone - Landline': {
        'regex': r'(?<!\d)(?:\(?\d{2,5}\)?[\s\-]?)\d{5,8}',
        'description': 'Landline numbers with area code',
        'category': 'Contact'
    },
    'GSTIN (India)': {
        'regex': r'(?<!\w)\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d](?!\w)',
        'description': 'Goods & Services Tax ID (India)',
        'category': 'Government ID'
    },
    'IP Address (IPv4)': {
        'regex': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'description': 'IPv4 address',
        'category': 'Network'
    },
    'IP Address (IPv6)': {
        'regex': r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',
        'description': 'IPv6 address',
        'category': 'Network'
    },
    'Postal/ZIP (Global)': {
        'regex': r'(?<!\w)[A-Z0-9\s\-]{3,10}(?!\w)',
        'description': 'ZIP/Postal code',
        'category': 'Address'
    },
    'PIN Code (India)': {
        'regex': r'(?<!\d)\d{6}(?!\d)',
        'description': '6-digit Indian PIN code',
        'category': 'Address'
    },
    'Date (ISO-8601)': {
        'regex': r'\b\d{4}-\d{2}-\d{2}\b',
        'description': 'ISO date format (YYYY-MM-DD)',
        'category': 'Date/Time'
    },
    'DateTime (ISO-8601)': {
        'regex': r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b',
        'description': 'ISO datetime format',
        'category': 'Date/Time'
    },
    'HTTP/HTTPS URL': {
        'regex': r'https?:\/\/[^\s]+',
        'description': 'Web URL',
        'category': 'Web'
    },
    'Multiple Spaces': {
        'regex': r'\s{2,}',
        'description': 'More than 1 consecutive space',
        'category': 'Formatting'
    },
    'Multiple Underscores': {
        'regex': r'_{2,}',
        'description': '2+ consecutive underscores',
        'category': 'Formatting'
    },
    'Non-ASCII Characters': {
        'regex': r'[^\x00-\x7F]',
        'description': 'Unicode characters (non-ASCII)',
        'category': 'Encoding'
    },
    'Special Characters': {
        'regex': r'[^\w\s]',
        'description': 'Non-alphanumeric characters',
        'category': 'Formatting'
    }
}

# Comprehensive special character patterns
SPECIAL_CHAR_PATTERNS = {
    'multiple_spaces': {'regex': r'\s+', 'display': 'multiple spaces', 'char': '  ', 'replacement': ' '},
    'multiple_underscores': {'regex': r'_+', 'display': 'multiple underscores', 'char': '__', 'replacement': '_'},
    '!': {'regex': r'!', 'display': 'exclamation mark', 'char': '!', 'replacement': '_'},
    '@': {'regex': r'@', 'display': 'at symbol', 'char': '@', 'replacement': '_at_'},
    '#': {'regex': r'#', 'display': 'hash', 'char': '#', 'replacement': '_hash_'},
    '$': {'regex': r'\$', 'display': 'dollar', 'char': '$', 'replacement': '_dollar_'},
    '%': {'regex': r'%', 'display': 'percent', 'char': '%', 'replacement': '_percent_'},
    '^': {'regex': r'\^', 'display': 'caret', 'char': '^', 'replacement': '_'},
    '&': {'regex': r'&', 'display': 'ampersand', 'char': '&', 'replacement': 'and'},
    '*': {'regex': r'\*', 'display': 'asterisk', 'char': '*', 'replacement': '_'},
    '(': {'regex': r'\(', 'display': 'left parenthesis', 'char': '(', 'replacement': '_'},
    ')': {'regex': r'\)', 'display': 'right parenthesis', 'char': ')', 'replacement': '_'},
    '-': {'regex': r'-', 'display': 'hyphen/minus', 'char': '-', 'replacement': '_'},
    '_': {'regex': r'_', 'display': 'underscore', 'char': '_', 'replacement': '_'},
    '+': {'regex': r'\+', 'display': 'plus', 'char': '+', 'replacement': '_'},
    '=': {'regex': r'=', 'display': 'equals', 'char': '=', 'replacement': '_'},
    '[': {'regex': r'\[', 'display': 'left bracket', 'char': '[', 'replacement': '_'},
    ']': {'regex': r'\]', 'display': 'right bracket', 'char': ']', 'replacement': '_'},
    '{': {'regex': r'\{', 'display': 'left brace', 'char': '{', 'replacement': '_'},
    '}': {'regex': r'\}', 'display': 'right brace', 'char': '}', 'replacement': '_'},
    '|': {'regex': r'\|', 'display': 'pipe', 'char': '|', 'replacement': '_'},
    '\\': {'regex': r'\\', 'display': 'backslash', 'char': '\\', 'replacement': '_'},
    ':': {'regex': r':', 'display': 'colon', 'char': ':', 'replacement': '_'},
    ';': {'regex': r';', 'display': 'semicolon', 'char': ';', 'replacement': '_'},
    '"': {'regex': r'"', 'display': 'double quote', 'char': '"', 'replacement': '_'},
    "'": {'regex': r"'", 'display': 'single quote', 'char': "'", 'replacement': '_'},
    '<': {'regex': r'<', 'display': 'less than', 'char': '<', 'replacement': '_'},
    '>': {'regex': r'>', 'display': 'greater than', 'char': '>', 'replacement': '_'},
    ',': {'regex': r',', 'display': 'comma', 'char': ',', 'replacement': '_'},
    '.': {'regex': r'\.', 'display': 'period', 'char': '.', 'replacement': '_'},
    '?': {'regex': r'\?', 'display': 'question mark', 'char': '?', 'replacement': '_'},
    '/': {'regex': r'/', 'display': 'forward slash', 'char': '/', 'replacement': '_'},
    '–': {'regex': r'–', 'display': 'en dash', 'char': '–', 'replacement': '_'},
    '—': {'regex': r'—', 'display': 'em dash', 'char': '—', 'replacement': '_'},
}


def safe_get_special_chars(prof):
    """Safely get special chars from profile object"""
    try:
        if hasattr(prof, 'special_chars') and prof.special_chars:
            chars = prof.special_chars
            if isinstance(chars, list) and len(chars) > 0:
                valid_chars = [c for c in chars if isinstance(c, dict) and 'count' in c]
                if valid_chars:
                    return valid_chars
        return []
    except Exception:
        return []


def format_char_display(char_info):
    """Format character for display"""
    try:
        symbol = char_info.get('symbol', '')
        name = char_info.get('display_name', char_info.get('name', 'unknown'))
        count = char_info.get('count', 0)
        
        if not symbol or count == 0:
            return None
        
        visual = "  " if symbol == '  ' else "__" if symbol == '__' else symbol
        return f"{visual} ({name}) appears {count:,} times"
    except Exception:
        return None


def generate_clean_regex_fixes(column_name, special_chars):
    """Generate clean, simple regex fixes"""
    fixes = []
    try:
        individual_patterns = []
        bulk_replacements = {}
        
        for char_info in special_chars:
            if not isinstance(char_info, dict): continue
            
            key = char_info.get('key', '')
            symbol = char_info.get('symbol', '')
            count = char_info.get('count', 0)
            
            if not symbol or count == 0: continue
            
            pattern_info = SPECIAL_CHAR_PATTERNS.get(key) or SPECIAL_CHAR_PATTERNS.get(symbol, {})
            regex_pattern = pattern_info.get('regex', re.escape(symbol))
            replacement = pattern_info.get('replacement', '_')
            
            if key == 'multiple_spaces':
                fixes.append({
                    'name': 'Remove Multiple Spaces',
                    'simple_regex': r'\s+',
                    'replacement': ' ',
                    'code': f"df['{column_name}'] = df['{column_name}'].str.replace(r'\\s+', ' ', regex=True)",
                    'description': 'Replace multiple consecutive spaces with single space'
                })
            elif key == 'multiple_underscores':
                fixes.append({
                    'name': 'Remove Multiple Underscores',
                    'simple_regex': r'_+',
                    'replacement': '_',
                    'code': f"df['{column_name}'] = df['{column_name}'].str.replace(r'_+', '_', regex=True)",
                    'description': 'Replace multiple consecutive underscores with single underscore'
                })
            else:
                individual_patterns.append(regex_pattern)
                bulk_replacements[regex_pattern] = replacement
        
        for pattern, repl in bulk_replacements.items():
            char = pattern.replace('\\', '') if pattern.startswith('\\') else pattern
            fixes.append({
                'name': f'Replace {char}',
                'simple_regex': pattern,
                'replacement': repl,
                'code': f"df['{column_name}'] = df['{column_name}'].str.replace(r'{pattern}', '{repl}', regex=True)",
                'description': f"Replace '{char}' with '{repl}'"
            })
        
        if len(individual_patterns) > 1:
            combined_pattern = '|'.join(individual_patterns[:6])
            fixes.insert(0, {
                'name': 'Bulk Replace All Special Chars',
                'simple_regex': combined_pattern,
                'replacement': '_',
                'code': f"df['{column_name}'] = df['{column_name}'].str.replace(r'{combined_pattern}', '_', regex=True)",
                'description': f'Replace {len(individual_patterns)} special characters with underscore'
            })
            
    except Exception as e:
        print(f"Error generating regex fixes: {e}")
    
    return fixes


def _remove_ghost_columns():
    """AUTO-FIX: Identifies and removes 'Unnamed' columns that are completely empty."""
    state = st.session_state.app_state
    if state.df is None:
        return False

    df = state.df
    ghost_cols = [
        col for col in df.columns 
        if str(col).startswith('Unnamed:') and df[col].isnull().all()
    ]

    if ghost_cols:
        state.df = df.drop(columns=ghost_cols)
        
        if hasattr(state, 'sheet_data') and hasattr(state, 'current_sheet_name'):
            current_sheet = getattr(state, 'current_sheet_name', None)
            if current_sheet and current_sheet in state.sheet_data:
                state.sheet_data[current_sheet] = state.df
        
        if state.column_profiles:
            for col in ghost_cols:
                state.column_profiles.pop(col, None)
                
        if state.quality_report:
            state.quality_report.total_columns = len(state.df.columns)
            if hasattr(state.quality_report, 'columns_with_issues'):
                state.quality_report.columns_with_issues = [
                    c for c in state.quality_report.columns_with_issues 
                    if c not in ghost_cols
                ]

        show_toast(f"🧹 Auto-cleaned {len(ghost_cols)} empty 'Unnamed' columns", "success")
        return True
    
    return False


# ==========================================
# NEW: PATTERN DETECTION FUNCTIONS
# ==========================================

def detect_patterns_in_dataframe(df):
    """
    Scan all text columns in dataframe for pattern matches.
    Returns list of pattern detection results.
    """
    pattern_results = []
    
    # Get object/string columns
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        col_data = df[col].dropna().astype(str)
        if col_data.empty:
            continue
            
        # Sample data for performance (first 1000 non-null values)
        sample_data = col_data.head(1000).tolist()
        
        for pattern_name, pattern_info in PATTERN_LIBRARY.items():
            regex = pattern_info['regex']
            matches = []
            match_count = 0
            unique_matches = set()
            
            for value in sample_data:
                found = re.findall(regex, value)
                if found:
                    match_count += len(found)
                    for match in found:
                        # Clean the match for display
                        clean_match = str(match).strip()[:50]
                        if clean_match:
                            unique_matches.add(clean_match)
            
            if match_count > 0:
                pattern_results.append({
                    'Column': col,
                    'Pattern Name': pattern_name,
                    'Category': pattern_info['category'],
                    'Description': pattern_info['description'],
                    'Regex': regex,
                    'Match Count': match_count,
                    'Unique Matches': len(unique_matches),
                    'Sample Matches': list(unique_matches)[:5],  # Top 5 unique samples
                    'Coverage %': round((match_count / len(sample_data)) * 100, 2),
                    'Variations': pattern_info.get('variations', [])
                })
    
    return pattern_results


def _analyze_column_patterns(df, column_name):
    """
    Analyze patterns for a specific column (used in column profiles).
    """
    col_data = df[column_name].dropna().astype(str)
    if col_data.empty:
        return []
    
    sample_data = col_data.head(500).tolist()
    detected = []
    
    for pattern_name, pattern_info in PATTERN_LIBRARY.items():
        regex = pattern_info['regex']
        match_count = 0
        
        for value in sample_data:
            if re.search(regex, value):
                match_count += 1
        
        if match_count > 0:
            coverage = round((match_count / len(sample_data)) * 100, 2)
            detected.append({
                'pattern': pattern_name,
                'category': pattern_info['category'],
                'coverage': coverage,
                'regex': regex
            })
    
    return sorted(detected, key=lambda x: x['coverage'], reverse=True)


def render_data_profiling():
    """Clean Data Profiling - Current Sheet Only"""
    
    state = st.session_state.app_state
    
    if state.df is None:
        st.info("📤 Please load data first in the 'Load Data' tab")
        return
    
    # Auto-Cleanup Ghost Columns
    if _remove_ghost_columns():
        time.sleep(0.5)
        st.rerun()

    # Check Profiling Status
    if not state.profiling_complete and state.processing_status == 'profiling':
        from state.session import check_profiling_complete
        if not check_profiling_complete():
            st.info(f"⏳ Profiling current data...")
            if state.upload_progress:
                prog = state.upload_progress
                msg = prog['message']
                if 'elapsed' in prog: msg += f" | Time: {prog['elapsed']}"
                if 'eta' in prog: msg += f" | ETA: {prog['eta']}"
                if 'rate' in prog: msg += f" ({prog['rate']})"
                st.progress(prog['percent'] / 100, text=msg)
            time.sleep(2)
            st.rerun()
            return
    
    # Render Dashboard
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-header">📊 Data Profiling</div>', unsafe_allow_html=True)
    
    _render_executive_summary()
    
    prof_tabs = st.tabs([
        "📋 Column Profiles", 
        "⚠️ Issues & Recommendations",
        "🔍 Pattern Detection",  # NEW TAB
        "📄 Export Report"
    ])
    
    with prof_tabs[0]:
        _render_column_profiles()
    
    with prof_tabs[1]:
        _render_issues_details()
    
    with prof_tabs[2]:
        _render_pattern_detection()  # NEW RENDER FUNCTION
    
    with prof_tabs[3]:
        _render_export_report()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_pattern_detection():
    """NEW: Render Pattern Detection Tab"""
    state = st.session_state.app_state
    
    st.markdown("### 🔍 Pattern Detection")
    st.markdown("Scanning data for common patterns (PII, IDs, Contact info, etc.)")
    
    with st.spinner("Analyzing patterns across all text columns..."):
        pattern_results = detect_patterns_in_dataframe(state.df)
    
    if not pattern_results:
        st.success("✅ No common patterns detected in the dataset")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patterns Found", len(pattern_results))
    with col2:
        categories = set(p['Category'] for p in pattern_results)
        st.metric("Categories", len(categories))
    with col3:
        columns_affected = set(p['Column'] for p in pattern_results)
        st.metric("Columns Affected", len(columns_affected))
    with col4:
        high_coverage = sum(1 for p in pattern_results if p['Coverage %'] > 50)
        st.metric("High Coverage (>50%)", high_coverage)
    
    st.divider()
    
    # Group by category
    by_category = defaultdict(list)
    for p in pattern_results:
        by_category[p['Category']].append(p)
    
    for category, patterns in sorted(by_category.items()):
        with st.expander(f"📂 {category} ({len(patterns)} patterns)", expanded=True):
            for p in patterns:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border-left: 4px solid #3b82f6;">
                            <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">
                                {p['Pattern Name']}
                            </div>
                            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 8px;">
                                {p['Description']}
                            </div>
                            <div style="font-size: 0.75rem; color: #94a3b8; font-family: monospace; background: #f1f5f9; padding: 4px 8px; border-radius: 4px;">
                                Regex: {p['Regex']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background: #f0fdf4; padding: 12px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #166534;">
                                {p['Match Count']}
                            </div>
                            <div style="font-size: 0.75rem; color: #64748b;">matches in</div>
                            <div style="font-size: 0.9rem; font-weight: 600; color: #1e293b;">
                                {p['Column']}
                            </div>
                            <div style="font-size: 0.75rem; color: #22c55e; margin-top: 4px;">
                                {p['Coverage %']}% coverage
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show sample matches
                    if p['Sample Matches']:
                        samples_html = " • ".join([f"<code>{s}</code>" for s in p['Sample Matches'][:3]])
                        st.markdown(f"<div style='margin-top: 8px; font-size: 0.8rem; color: #64748b;'>Samples: {samples_html}</div>", 
                                   unsafe_allow_html=True)
                    
                    # Show variations if available
                    if p['Variations']:
                        with st.expander("📝 Accepted Variations", expanded=False):
                            var_html = "<br>".join([f"• <code>{v}</code>" for v in p['Variations']])
                            st.markdown(var_html, unsafe_allow_html=True)
                    
                    st.divider()


def _calculate_overall_quality_score(profiles, total_rows):
    """
    Calculate Overall Quality Score based on:
    - Completeness (40%): Average of non-null percentages across all columns
    - Uniqueness (30%): Average uniqueness ratio
    - Consistency (20%): Based on formatting consistency and pattern adherence
    - Validity (10%): Based on business rule violations and special chars
    """
    if not profiles:
        return 0
    
    scores = []
    
    for prof in profiles.values():
        # Completeness score (0-100)
        completeness = getattr(prof, 'non_null_percentage', 100)
        
        # Uniqueness score (0-100) - penalize both too low and too high (potential primary keys)
        unique_pct = prof.unique_percentage
        if unique_pct == 100 and prof.total_rows > 10:
            uniqueness = 90  # Slight penalty for potential PK (might be ID column)
        else:
            uniqueness = min(100, unique_pct * 1.2)  # Boost uniqueness
        
        # Consistency score (0-100)
        consistency = 100
        if hasattr(prof, 'formatting_info') and prof.formatting_info:
            if not prof.formatting_info.get('consistent_case', True):
                consistency -= 20
        if getattr(prof, 'out_of_bounds_count', 0) > 0:
            consistency -= min(30, prof.out_of_bounds_count / prof.total_rows * 100)
        
        # Validity score (0-100)
        validity = 100
        if hasattr(prof, 'business_rule_violations') and prof.business_rule_violations:
            validity -= min(50, len(prof.business_rule_violations) * 10)
        special_chars = safe_get_special_chars(prof)
        if special_chars:
            validity -= min(20, len(special_chars) * 2)
        
        # Weighted average for this column
        col_score = (completeness * 0.4 + uniqueness * 0.3 + 
                    consistency * 0.2 + validity * 0.1)
        scores.append(col_score)
    
    # Overall score is average of all column scores
    return round(sum(scores) / len(scores), 1)


def _calculate_overall_completeness_score(profiles):
    """
    Calculate Overall Completeness Score:
    Average of (non-null count / total rows) across all columns * 100
    """
    if not profiles:
        return 0
    
    total_cells = sum(prof.total_rows for prof in profiles.values())
    missing_cells = sum(prof.null_count for prof in profiles.values())
    
    if total_cells == 0:
        return 100
    
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    return round(completeness, 2)


def _get_duplicate_analysis(df):
    """
    Analyze duplicate rows and return detailed information.
    Returns: (duplicate_count, duplicate_details_list)
    """
    # Find exact duplicates
    duplicates = df[df.duplicated(keep=False)]
    
    if len(duplicates) == 0:
        return 0, []
    
    # Group by all columns to find duplicate groups
    duplicate_groups = duplicates.groupby(list(df.columns)).size().reset_index(name='count')
    duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1]
    
    details = []
    for idx, row in duplicate_groups.head(10).iterrows():  # Top 10 duplicate patterns
        count = row['count']
        # Get columns that vary (should be none for exact duplicates)
        sample_cols = {k: v for k, v in row.items() if k != 'count' and pd.notna(v)}
        sample_str = ' | '.join([f"{k}={str(v)[:30]}" for k, v in list(sample_cols.items())[:3]])
        
        details.append({
            'count': count,
            'occurrences': count,
            'sample': sample_str,
            'all_columns_match': 'All columns identical'
        })
    
    total_duplicates = len(duplicates)
    return total_duplicates, details


def _get_columns_with_issues_detailed(profiles):
    """
    Get detailed list of columns with issues including reasons.
    """
    issues = []
    
    for col_name, prof in profiles.items():
        reasons = []
        severity = "Low"
        
        # Check null percentage
        if prof.null_percentage > 50:
            reasons.append(f"High null rate: {prof.null_percentage:.1f}%")
            severity = "Critical"
        elif prof.null_percentage > 20:
            reasons.append(f"Moderate null rate: {prof.null_percentage:.1f}%")
            if severity != "Critical":
                severity = "High"
        elif prof.null_percentage > 5:
            reasons.append(f"Low null rate: {prof.null_percentage:.1f}%")
        
        # Check special characters
        special_chars = safe_get_special_chars(prof)
        if special_chars:
            char_types = len(special_chars)
            total_instances = sum(c.get('count', 0) for c in special_chars)
            reasons.append(f"Special characters: {char_types} types, {total_instances} instances")
            if severity == "Low":
                severity = "Medium"
        
        # Check business rules
        if hasattr(prof, 'business_rule_violations') and prof.business_rule_violations:
            violations = len(prof.business_rule_violations)
            reasons.append(f"Business rule violations: {violations}")
            severity = "Critical" if violations > 2 else "High"
        
        # Check formatting consistency
        if hasattr(prof, 'formatting_info') and prof.formatting_info:
            if not prof.formatting_info.get('consistent_case', True):
                reasons.append("Inconsistent case formatting")
        
        # Check out of bounds
        if getattr(prof, 'out_of_bounds_count', 0) > 0:
            reasons.append(f"Out of bounds values: {prof.out_of_bounds_count}")
        
        if reasons:
            issues.append({
                'column': col_name,
                'severity': severity,
                'reasons': '; '.join(reasons),
                'issue_count': len(reasons)
            })
    
    # Sort by severity
    severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    issues.sort(key=lambda x: severity_order.get(x['severity'], 4))
    
    return issues


def _render_executive_summary():
    """Render executive summary - ENHANCED version (removed duplicate analysis)"""
    state = st.session_state.app_state
    report = state.quality_report
    
    if not report:
        st.warning("Quality report not available")
        return
    
    # Calculate refined scores
    overall_quality = _calculate_overall_quality_score(state.column_profiles, len(state.df))
    overall_completeness = _calculate_overall_completeness_score(state.column_profiles)
    
    status = "Excellent" if overall_quality >= 90 else "Good" if overall_quality >= 75 else "Fair" if overall_quality >= 60 else "Critical"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                    padding: 16px; border-radius: 8px; color: white; text-align: center;">
            <div style="font-size: 2rem; font-weight: bold;">{overall_quality:.0f}</div>
            <div style="font-size: 0.9rem;">{status}</div>
            <div style="font-size: 0.7rem; margin-top: 4px; opacity: 0.8;">Quality Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Total Rows", f"{len(state.df):,}")
    
    with col3:
        st.metric("Total Columns", len(state.df.columns))
    
    with col4:
        # Calculate high risk columns
        high_risk_count = sum(
            1 for prof in state.column_profiles.values()
            if getattr(prof, 'risk_level', 'Low') == 'High'
        )
        st.metric("High Risk Columns", high_risk_count)
    
    # Second row of metrics - ENHANCED (removed duplicate rows and high risk columns)
    col1, col2, col3 = st.columns(3)  # Changed from 4 to 3 columns
    
    with col1:
        st.metric("Overall Completeness", f"{overall_completeness:.1f}%")
    
    with col2:
        # Missing cells count (not percentage)
        total_missing = sum(prof.null_count for prof in state.column_profiles.values())
        st.metric("Missing Cells", f"{total_missing:,}")
    
    with col3:
        # Columns with issues count
        issues = _get_columns_with_issues_detailed(state.column_profiles)
        st.metric("Columns with Issues", len(issues))
    
    # REMOVED: Duplicate Rows metric and High Risk Columns from second row


def _render_column_profiles():
    """Clean column profiles - Refined version"""
    state = st.session_state.app_state
    profiles = state.column_profiles
    
    if not profiles:
        st.warning("No column profiles available")
        return
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("🔍 Search columns:", key="prof_search")
    with col2:
        filter_type = st.selectbox("Filter:", 
                                  ["All", "Numeric", "Text", "Date", "High Risk"],
                                  key="prof_filter")
    with col3:
        sort_by = st.selectbox("Sort:", 
                              ["Name", "Null %", "Risk Score"],
                              key="prof_sort")
    
    filtered = profiles.copy()
    
    if search:
        filtered = {k: v for k, v in filtered.items() if search.lower() in k.lower()}
    
    if filter_type == "Numeric":
        filtered = {k: v for k, v in filtered.items() 
                   if any(t in v.dtype for t in ['int', 'float'])}
    elif filter_type == "Text":
        filtered = {k: v for k, v in filtered.items() if v.dtype == 'object'}
    elif filter_type == "Date":
        filtered = {k: v for k, v in filtered.items() if 'date' in v.dtype}
    elif filter_type == "High Risk":
        filtered = {k: v for k, v in filtered.items() 
                   if getattr(v, 'risk_level', 'Low') == 'High'}
    
    if sort_by == "Null %":
        filtered = dict(sorted(filtered.items(), 
                             key=lambda x: x[1].null_percentage, reverse=True))
    elif sort_by == "Risk Score":
        filtered = dict(sorted(filtered.items(), 
                             key=lambda x: getattr(x[1], 'risk_score', 0), reverse=True))
    
    st.write(f"**Showing {len(filtered)} of {len(profiles)} columns**")
    
    MAX_DISPLAY = 50
    items_to_display = list(filtered.items())[:MAX_DISPLAY]
    
    if len(filtered) > MAX_DISPLAY:
        st.caption(f"⚠️ Displaying top {MAX_DISPLAY} columns for performance. Use filters to see specific columns.")

    for col_name, prof in items_to_display:
        risk_level = getattr(prof, 'risk_level', 'Low')
        risk_score = getattr(prof, 'risk_score', 0)
        
        with st.expander(f"**{col_name}**"):
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([  # Added tab5 for Patterns
                "📊 Overview", "🔍 Data Quality", "📏 Patterns & Format", 
                "🔎 Pattern Matches", "🛠️ Action Items"  # NEW TAB
            ])
            
            with tab1:
                _render_overview_tab(prof)
            
            with tab2:
                _render_data_quality_tab(prof, risk_level, risk_score)
            
            with tab3:
                _render_patterns_tab(prof)
            
            with tab4:
                _render_column_pattern_matches(col_name)  # NEW
            
            with tab5:
                _render_action_items_tab(col_name, prof)


def _render_column_pattern_matches(col_name):
    """NEW: Show pattern matches for specific column"""
    state = st.session_state.app_state
    
    patterns = _analyze_column_patterns(state.df, col_name)
    
    if not patterns:
        st.info("No standard patterns detected in this column")
        return
    
    st.markdown(f"### 🔍 Detected Patterns in `{col_name}`")
    
    for p in patterns:
        coverage_color = "#22c55e" if p['coverage'] > 50 else "#f59e0b" if p['coverage'] > 20 else "#ef4444"
        
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 12px; border-radius: 6px; border-left: 4px solid {coverage_color}; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-weight: 600; color: #1e293b;">{p['pattern']}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">{p['category']} • {p['regex']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: {coverage_color};">{p['coverage']}%</div>
                    <div style="font-size: 0.7rem; color: #64748b;">coverage</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_overview_tab(prof):
    """Render Overview tab content"""
    completeness = getattr(prof, 'non_null_percentage', 0)
    completeness_color = "#22c55e" if completeness > 90 else "#f59e0b" if completeness > 50 else "#ef4444"
    
    st.markdown(f"""
    <div style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #3b82f6;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0;">
            <div style="display: flex; gap: 20px; align-items: center;">
                <span style="font-size: 0.75rem; color: #64748b; font-weight: 500;">📌 DATA TYPE</span>
                <span style="font-size: 0.9rem; font-weight: 600; color: #1e293b;">{getattr(prof, 'human_readable_dtype', 'Unknown')}</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0;">
            <div style="display: flex; gap: 16px; align-items: center; flex: 1;">
                <span style="font-size: 0.75rem; color: #64748b; font-weight: 500;">📊 COMPLETENESS</span>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <span style="font-size: 0.85rem; color: #1e293b;">
                        <span style="font-weight: 600;">{getattr(prof, 'non_null_count', 0):,}</span> 
                        <span style="color: #94a3b8; font-size: 0.75rem;">non-null</span>
                    </span>
                    <span style="color: #cbd5e1;">•</span>
                    <span style="font-size: 0.85rem; color: #1e293b;">
                        <span style="font-weight: 600;">{prof.null_count:,}</span> 
                        <span style="color: #94a3b8; font-size: 0.75rem;">nulls</span>
                    </span>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="background: #e5e7eb; border-radius: 3px; height: 6px; width: 60px; overflow: hidden;">
                    <div style="background: {completeness_color}; width: {completeness}%; height: 100%;"></div>
                </div>
                <span style="font-size: 0.85rem; font-weight: 600; color: {completeness_color};">{completeness:.1f}%</span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; gap: 16px; align-items: center;">
                <span style="font-size: 0.75rem; color: #64748b; font-weight: 500;">🔑 UNIQUENESS</span>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <span style="font-size: 0.85rem; color: #1e293b;">
                        <span style="font-weight: 600;">{prof.unique_count:,}</span> 
                        <span style="color: #94a3b8; font-size: 0.75rem;">unique</span>
                    </span>
                    <span style="color: #cbd5e1;">•</span>
                    <span style="font-size: 0.85rem; color: #1e293b;">
                        <span style="font-weight: 600;">{prof.unique_percentage:.1f}%</span> 
                        <span style="color: #94a3b8; font-size: 0.75rem;">unique</span>
                    </span>
                    <span style="color: #cbd5e1;">•</span>
                    <span style="font-size: 0.85rem; color: #1e293b;">
                        <span style="font-weight: 600;">{getattr(prof, 'duplicate_percentage', 0):.1f}%</span> 
                        <span style="color: #94a3b8; font-size: 0.75rem;">duplicate</span>
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_data_quality_tab(prof, risk_level, risk_score):
    """Render Data Quality tab content"""
    risk_bg = "#fee2e2" if risk_level == "High" else "#fed7aa" if risk_level == "Medium" else "#d1fae5"
    risk_text = "#991b1b" if risk_level == "High" else "#9a3412" if risk_level == "Medium" else "#065f46"
    
    st.markdown(f"""
    <div style="background: #f8fafc; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px;">
        <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.7rem; color: #64748b; font-weight: 600; letter-spacing: 0.5px;">RISK</span>
                <span style="padding: 2px 8px; background: {risk_bg}; color: {risk_text}; border-radius: 3px; font-size: 0.75rem; font-weight: 600;">{risk_level}</span>
                <span style="font-size: 0.75rem; color: #64748b;">({risk_score}/100)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if hasattr(prof, 'key_issues') and prof.key_issues:
        st.markdown("""
        <div style="background: #fef2f2; padding: 6px 10px; border-radius: 4px; border-left: 2px solid #ef4444; margin-bottom: 8px;">
            <div style="font-size: 0.7rem; color: #991b1b; font-weight: 600; margin-bottom: 4px; letter-spacing: 0.3px;">⚠️ KEY ISSUES</div>
        """, unsafe_allow_html=True)
        for issue in prof.key_issues:
            st.markdown(f"<div style='font-size: 0.8rem; color: #1e293b; margin: 2px 0; padding-left: 8px;'>• {issue}</div>", 
                       unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if hasattr(prof, 'business_rule_violations') and prof.business_rule_violations:
        st.markdown("""
        <div style="background: #fef2f2; padding: 6px 10px; border-radius: 4px; border-left: 2px solid #dc2626; margin-bottom: 8px;">
            <div style="font-size: 0.7rem; color: #991b1b; font-weight: 600; margin-bottom: 4px; letter-spacing: 0.3px;">🚫 BUSINESS RULE VIOLATIONS</div>
        """, unsafe_allow_html=True)
        for violation in prof.business_rule_violations:
            st.markdown(f"<div style='font-size: 0.8rem; color: #1e293b; margin: 2px 0; padding-left: 8px;'>• {violation}</div>", 
                       unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if prof.special_chars:
        displayed_chars = []
        for c in prof.special_chars[:10]:
            try:
                display = format_char_display(c)
                if display: displayed_chars.append(display)
            except: pass
        
        if displayed_chars:
            total_rows = getattr(prof, 'total_special_char_rows', 0)
            st.markdown(f"""
            <div style="background: #fffbeb; padding: 6px 10px; border-radius: 4px; border-left: 2px solid #f59e0b; margin-bottom: 8px;">
                <div style="font-size: 0.7rem; color: #92400e; font-weight: 600; margin-bottom: 4px; letter-spacing: 0.3px;">⚡ SPECIAL CHARACTERS ({total_rows:,} rows affected)</div>
            """, unsafe_allow_html=True)
            for dc in displayed_chars:
                st.markdown(f"<div style='font-size: 0.75rem; color: #1e293b; margin: 1px 0; padding-left: 8px; font-family: monospace;'>• {dc}</div>", 
                           unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    if not (hasattr(prof, 'key_issues') and prof.key_issues) and \
       not (hasattr(prof, 'business_rule_violations') and prof.business_rule_violations) and \
       not prof.special_chars:
        st.markdown("""
        <div style="background: #f0fdf4; padding: 8px 12px; border-radius: 4px; border-left: 2px solid #22c55e; text-align: center;">
            <span style="font-size: 0.85rem; color: #166534; font-weight: 500;">✅ No data quality issues detected</span>
        </div>
        """, unsafe_allow_html=True)


def _render_patterns_tab(prof):
    """Render Patterns tab"""
    min_len = getattr(prof, 'min_length', 0)
    max_len = getattr(prof, 'max_length', 0)
    avg_len = getattr(prof, 'avg_length', 0)
    out_of_bounds = getattr(prof, 'out_of_bounds_count', 0)
    oob_color = "#ef4444" if out_of_bounds > 0 else "#1e293b"
    
    st.markdown(f"""
    <div style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #8b5cf6; margin-bottom: 12px;">
        <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; margin-bottom: 8px; letter-spacing: 0.5px;">📏 LENGTH ANALYSIS</div>
        <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; gap: 6px; align-items: baseline;">
                <span style="font-size: 0.7rem; color: #94a3b8;">MIN</span>
                <span style="font-size: 0.9rem; font-weight: 600; color: #1e293b;">{min_len}</span>
            </div>
            <span style="color: #cbd5e1;">•</span>
            <div style="display: flex; gap: 6px; align-items: baseline;">
                <span style="font-size: 0.7rem; color: #94a3b8;">MAX</span>
                <span style="font-size: 0.9rem; font-weight: 600; color: #1e293b;">{max_len}</span>
            </div>
            <span style="color: #cbd5e1;">•</span>
            <div style="display: flex; gap: 6px; align-items: baseline;">
                <span style="font-size: 0.7rem; color: #94a3b8;">AVG</span>
                <span style="font-size: 0.9rem; font-weight: 600; color: #1e293b;">{avg_len:.1f}</span>
            </div>
            <span style="color: #cbd5e1;">•</span>
            <div style="display: flex; gap: 6px; align-items: baseline;">
                <span style="font-size: 0.7rem; color: #94a3b8;">OUT OF BOUNDS</span>
                <span style="font-size: 0.9rem; font-weight: 600; color: {oob_color};">{out_of_bounds}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    patterns_list = []
    if hasattr(prof, 'pattern_counts') and prof.pattern_counts:
        patterns_list = list(prof.pattern_counts.keys())
    elif hasattr(prof, 'patterns') and prof.patterns.get('pattern_types'):
        patterns_list = prof.patterns['pattern_types']
    
    if patterns_list:
        patterns_html = "".join([
            f'<span style="padding: 2px 8px; background: #dbeafe; color: #1e40af; border-radius: 3px; font-size: 0.75rem; font-weight: 600; margin-right: 6px; margin-bottom: 4px; display: inline-block;">{p}</span>'
            for p in patterns_list
        ])
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #3b82f6; margin-bottom: 12px;">
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; margin-bottom: 8px; letter-spacing: 0.5px;">🔍 PATTERNS DETECTED</div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                {patterns_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #3b82f6; margin-bottom: 12px;">
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; margin-bottom: 8px; letter-spacing: 0.5px;">🔍 PATTERNS DETECTED</div>
            <div style="text-align: center; padding: 8px;">
                <span style="font-size: 0.8rem; color: #94a3b8;">No specific patterns detected</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if hasattr(prof, 'formatting_info') and prof.formatting_info:
        fmt = prof.formatting_info
        consistent = fmt.get('consistent_case', False)
        case_type = fmt.get('case_type', '')
        
        consistency_bg = "#dcfce7" if consistent else "#fef3c7"
        consistency_text = "#166534" if consistent else "#92400e"
        status_text = "✅ CONSISTENT" if consistent else "❌ INCONSISTENT"
        
        consistency_badge = f'<span style="padding: 2px 8px; background: {consistency_bg}; color: {consistency_text}; border-radius: 3px; font-size: 0.75rem; font-weight: 600;">{status_text}</span>'
        case_badge_html = ""
        if case_type:
            case_badge_html = f'<span style="color: #cbd5e1;">•</span><div style="display: flex; gap: 6px; align-items: center;"><span style="font-size: 0.7rem; color: #94a3b8;">DOMINANT CASE</span><span style="font-size: 0.8rem; font-weight: 600; color: #1e293b;">{case_type}</span></div>'
        
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #10b981;">
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; margin-bottom: 8px; letter-spacing: 0.5px;">🎨 FORMATTING CONSISTENCY</div>
            <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                <div style="display: flex; gap: 6px; align-items: center;">
                    <span style="font-size: 0.7rem; color: #94a3b8;">CASE CONSISTENCY</span>
                    {consistency_badge}
                </div>
                {case_badge_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_action_items_tab(col_name, prof):
    """Render Action Items"""
    has_recs = hasattr(prof, 'cleansing_recommendations') and prof.cleansing_recommendations
    
    if has_recs:
        st.markdown("""
        <div style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #22c55e; margin-bottom: 12px;">
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; margin-bottom: 8px; letter-spacing: 0.5px;">🛠️ RECOMMENDED ACTIONS</div>
        """, unsafe_allow_html=True)
        
        for idx, rec in enumerate(prof.cleansing_recommendations[:3], 1):
            st.markdown(f"""
            <div style="background: #f0fdf4; padding: 8px 12px; border-radius: 4px; border-left: 2px solid #22c55e; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-size: 0.8rem; font-weight: 600; color: #15803d;">#{idx} {rec['action']}</span>
                    <span style="padding: 1px 6px; background: #dcfce7; color: #166534; border-radius: 2px; font-size: 0.65rem; font-weight: 600;">{rec['impact']}</span>
                </div>
                <div style="font-size: 0.75rem; color: #374151; line-height: 1.4;">{rec['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if len(prof.cleansing_recommendations) > 3:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 8px;">
                <span style="font-size: 0.7rem; color: #64748b;">+{len(prof.cleansing_recommendations) - 3} more recommendations available</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #f0fdf4; padding: 12px; border-radius: 6px; border-left: 3px solid #22c55e; text-align: center; margin-bottom: 12px;">
            <span style="font-size: 0.85rem; color: #166534; font-weight: 500;">✅ No cleansing actions required</span>
        </div>
        """, unsafe_allow_html=True)
    
    if prof.special_chars:
        fixes = generate_clean_regex_fixes(col_name, prof.special_chars)
        if fixes:
            st.markdown("""
            <div style="background: #f8fafc; padding: 10px 14px; border-radius: 6px; border-left: 3px solid #f59e0b; margin-bottom: 12px;">
                <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; margin-bottom: 8px; letter-spacing: 0.5px;">⚡ REGEX QUICK FIXES</div>
            """, unsafe_allow_html=True)
            
            for fix in fixes[:3]:
                fix_name = fix.get('name', 'Unknown Fix')
                fix_description = fix.get('description', 'No description available')
                fix_code = fix.get('code', '# No code available')
                
                st.markdown(f"""
                <div style="background: #fffbeb; padding: 10px; border-radius: 4px; border-left: 2px solid #f59e0b; margin-bottom: 8px;">
                    <div style="font-size: 0.8rem; font-weight: 600; color: #92400e; margin-bottom: 6px;">{fix_name}</div>
                    <div style="font-size: 0.7rem; color: #78716c; margin-bottom: 6px;">{fix_description}</div>
                </div>
                """, unsafe_allow_html=True)
                st.code(fix_code, language='python')
            
            st.markdown("</div>", unsafe_allow_html=True)


def _render_issues_details():
    """Detailed issues view"""
    state = st.session_state.app_state
    st.markdown("### ⚠️ Data Quality Issues Summary")
    
    all_issues = []
    
    for col_name, prof in state.column_profiles.items():
        try:
            col_issues = []
            if prof.null_percentage > 50:
                col_issues.append(("critical", f"🚫 {prof.null_percentage:.1f}% missing values"))
            elif prof.null_percentage > 10:
                col_issues.append(("warning", f"⚠️ {prof.null_percentage:.1f}% missing values"))
            
            if hasattr(prof, 'blank_count'):
                empty = prof.blank_count.get('empty_string_count', 0)
                whitespace = prof.blank_count.get('whitespace_only_count', 0)
                if empty > 0: col_issues.append(("warning", f"📄 {empty} empty strings"))
                if whitespace > 0: col_issues.append(("info", f"␣ {whitespace} whitespace-only values"))
            
            for char_info in safe_get_special_chars(prof):
                try:
                    display_text = format_char_display(char_info)
                    if display_text: col_issues.append(("warning", f"⚡ {display_text}"))
                except Exception: continue
            
            if col_issues:
                all_issues.append({
                    'column': col_name,
                    'dtype': prof.dtype,
                    'issues': col_issues
                })
        except Exception: continue
    
    critical_issues = [x for x in all_issues if any(i[0] == 'critical' for i in x['issues'])]
    warning_issues = [x for x in all_issues if any(i[0] == 'warning' for i in x['issues']) and x not in critical_issues]
    info_issues = [x for x in all_issues if x not in critical_issues and x not in warning_issues]
    
    if critical_issues:
        st.error(f"🔴 Critical Issues ({len(critical_issues)} columns)")
        for item in critical_issues:
            issues_html = "".join([f'<li>{issue[1]}</li>' for issue in item['issues']])
            st.markdown(f"""
            <div style="border-left: 4px solid #ef4444; padding: 12px 16px; margin: 8px 0; background: #fef2f2; border-radius: 0 8px 8px 0;">
                <b>{item['column']}</b> <span style="color: #6b7280;">({item['dtype']})</span>
                <ul style="margin: 8px 0; padding-left: 20px;">{issues_html}</ul>
            </div>
            """, unsafe_allow_html=True)
    
    if warning_issues:
        st.warning(f"🟡 Warnings ({len(warning_issues)} columns)")
        for item in warning_issues:
            issues_html = "".join([f'<li>{issue[1]}</li>' for issue in item['issues']])
            st.markdown(f"""
            <div style="border-left: 4px solid #f59e0b; padding: 12px 16px; margin: 8px 0; background: #fffbeb; border-radius: 0 8px 8px 0;">
                <b>{item['column']}</b> <span style="color: #6b7280;">({item['dtype']})</span>
                <ul style="margin: 8px 0; padding-left: 20px;">{issues_html}</ul>
            </div>
            """, unsafe_allow_html=True)
    
    if info_issues:
        st.info(f"ℹ️ Minor Issues ({len(info_issues)} columns)")
        for item in info_issues:
            issues_html = "".join([f'<li>{issue[1]}</li>' for issue in item['issues']])
            st.markdown(f"""
            <div style="border-left: 4px solid #3b82f6; padding: 12px 16px; margin: 8px 0; background: #eff6ff; border-radius: 0 8px 8px 0;">
                <b>{item['column']}</b> <span style="color: #6b7280;">({item['dtype']})</span>
                <ul style="margin: 8px 0; padding-left: 20px;">{issues_html}</ul>
            </div>
            """, unsafe_allow_html=True)
    
    if not all_issues:
        st.success("✅ No issues found! All columns look clean.")
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Critical", len(critical_issues))
    with col2: st.metric("Warnings", len(warning_issues))
    with col3: st.metric("Minor", len(info_issues))


def _render_export_report():
    """Export profiling report"""
    state = st.session_state.app_state
    
    st.markdown(f"### 📄 Export Profiling Report")
    
    report_format = st.radio("Format:", ["Excel", "JSON"], horizontal=True)
    
    if report_format == "Excel":
        if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
            _generate_excel_report()
    else:
        if st.button("📋 Generate JSON Report", type="primary", use_container_width=True):
            _generate_json_report()


def _is_non_ascii_char(char):
    """Check if character is a non-ASCII character that should be flagged."""
    if char.isspace():
        return False
    
    ord_val = ord(char)
    
    if ord_val <= 127:
        return False
    
    return True


def _get_script_category(char):
    """Categorize non-ASCII characters by their Unicode script."""
    try:
        name = unicodedata.name(char, 'UNKNOWN')
        
        if 'CJK' in name or any('\u4e00' <= char <= '\u9fff' for _ in [0]):
            return 'CJK (Chinese/Japanese/Korean)'
        elif any('\u3040' <= char <= '\u309f' for _ in [0]):
            return 'Japanese (Hiragana)'
        elif any('\u30a0' <= char <= '\u30ff' for _ in [0]):
            return 'Japanese (Katakana)'
        elif any('\uac00' <= char <= '\ud7af' for _ in [0]):
            return 'Korean (Hangul)'
        elif any('\u0370' <= char <= '\u03ff' for _ in [0]):
            return 'Greek'
        elif any('\u0400' <= char <= '\u04ff' for _ in [0]):
            return 'Cyrillic'
        elif any('\u0600' <= char <= '\u06ff' for _ in [0]):
            return 'Arabic'
        elif any('\u0900' <= char <= '\u097f' for _ in [0]):
            return 'Devanagari (Hindi/Sanskrit)'
        elif 'LATIN' in name:
            if any(c in name for c in ['WITH GRAVE', 'WITH ACUTE', 'WITH CIRCUMFLEX', 'WITH TILDE', 'WITH DIAERESIS', 'WITH RING ABOVE']):
                return 'Latin (Accented)'
            return 'Latin Extended'
        else:
            return 'Other Unicode'
    except:
        return 'Unknown'


def _analyze_special_chars_detailed(df):
    """Detailed special character analysis."""
    special_chars_data = []
    
    object_cols = df.select_dtypes(include=['object']).columns
    
    for col in object_cols:
        char_counter = Counter()
        char_examples = defaultdict(list)
        
        for val in df[col].dropna().astype(str):
            for char in set(val):
                is_special = False
                
                if _is_non_ascii_char(char):
                    is_special = True
                elif not char.isalnum() and not char.isspace():
                    is_special = True
                
                if is_special:
                    char_counter[char] += val.count(char)
                    if len(char_examples[char]) < 3:
                        char_examples[char].append(val[:50])
        
        for char, count in char_counter.most_common():
            try:
                uname = unicodedata.name(char)
            except ValueError:
                uname = "UNKNOWN"
            
            is_superscript = "SUPERSCRIPT" in uname
            is_currency = any(x in uname for x in ['DOLLAR', 'EURO', 'POUND', 'YEN', 'CURRENCY'])
            is_math = any(x in uname for x in ['PLUS', 'MINUS', 'EQUALS', 'MULTIPLY', 'DIVIDE', 'PERCENT'])
            is_punctuation = any(x in uname for x in ['COMMA', 'PERIOD', 'COLON', 'SEMICOLON', 'EXCLAMATION', 'QUESTION'])
            
            script_category = _get_script_category(char) if ord(char) > 127 else 'ASCII Symbol'
            
            special_chars_data.append({
                'Column': col,
                'Character': char,
                'Unicode_Name': uname,
                'Script_Category': script_category,
                'Hex': hex(ord(char)),
                'Count': count,
                'Is_Superscript': is_superscript,
                'Is_Currency': is_currency,
                'Is_Math': is_math,
                'Is_Punctuation': is_punctuation,
                'Example_Values': ' | '.join(char_examples.get(char, [])[:3])
            })
    
    return special_chars_data


def _generate_excel_report():
    """Generate ENHANCED Excel report with Pattern Found sheet"""
    state = st.session_state.app_state
    progress_bar = st.progress(0, text="Initializing report...")
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Calculate metrics
            overall_quality = _calculate_overall_quality_score(state.column_profiles, len(state.df))
            overall_completeness = _calculate_overall_completeness_score(state.column_profiles)
            total_missing = sum(prof.null_count for prof in state.column_profiles.values())
            dup_count, dup_details = _get_duplicate_analysis(state.df)
            columns_with_issues = _get_columns_with_issues_detailed(state.column_profiles)
            
            # 1. Executive Summary (ENHANCED - removed duplicate analysis and high risk columns)
            progress_bar.progress(10, text="Generating Executive Summary...")
            try:
                overview_data = {
                    'Metric': [
                        'Total Rows',
                        'Total Columns',
                        'Overall Quality Score',
                        'Overall Quality Score Formula',
                        'Overall Completeness Score',
                        'Overall Completeness Score Formula',
                        'Missing Cells (Count)',
                        'Generated At'
                        # REMOVED: Duplicate Rows, Duplicate Analysis, High Risk Columns
                    ],
                    'Value': [
                        len(state.df),
                        len(state.df.columns),
                        f"{overall_quality:.1f}",
                        "Weighted: Completeness(40%) + Uniqueness(30%) + Consistency(20%) + Validity(10%)",
                        f"{overall_completeness:.2f}%",
                        "(Total Cells - Missing Cells) / Total Cells * 100",
                        total_missing,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                pd.DataFrame(overview_data).to_excel(writer, sheet_name='Executive Summary', index=False)
            except Exception as e:
                pd.DataFrame({'Error': [f'Could not generate overview: {str(e)}']}).to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # 2. Column Profiles (Refined)
            progress_bar.progress(25, text="Generating Column Profiles...")
            try:
                profile_data = []
                total_cols = len(state.column_profiles)
                
                for idx, (col_name, prof) in enumerate(state.column_profiles.items()):
                    if idx % 10 == 0:
                        progress_bar.progress(25 + int((idx / total_cols) * 15), text=f"Processing: {col_name}")
                    
                    # Calculate duplicate count for this column
                    dup_count_col = prof.total_rows - prof.unique_count
                    
                    profile_data.append({
                        'Column': col_name,
                        'Data Type (Human Readable)': getattr(prof, 'human_readable_dtype', 'Unknown'),
                        'Data Type (Technical)': prof.dtype,
                        'Risk Level': getattr(prof, 'risk_level', 'Low'),
                        'Risk Score': getattr(prof, 'risk_score', 0),
                        'Total Rows': prof.total_rows,
                        'Non-Null Count': getattr(prof, 'non_null_count', 0),
                        'Null Count': prof.null_count,
                        'Null %': round(prof.null_percentage, 2),
                        'Unique Count': prof.unique_count,
                        'Duplicate Count': dup_count_col,
                        'Unique %': round(prof.unique_percentage, 2),
                        'Duplicate %': round(getattr(prof, 'duplicate_percentage', 0), 2),
                        'Min Length': getattr(prof, 'min_length', 0),
                        'Max Length': getattr(prof, 'max_length', 0),
                        'Avg Length': getattr(prof, 'avg_length', 0),
                        'Out of Bounds Lengths': getattr(prof, 'out_of_bounds_count', 0)
                    })
                pd.DataFrame(profile_data).to_excel(writer, sheet_name='Column Profiles', index=False)
            except Exception as e:
                print(f"Error generating profiles: {e}")
                pd.DataFrame({'Error': ['Could not generate column profiles']}).to_excel(writer, sheet_name='Column Profiles', index=False)
            
            # 3. Special Characters (unchanged functionality)
            progress_bar.progress(45, text="Analyzing Special Characters...")
            try:
                special_chars_data = _analyze_special_chars_detailed(state.df)
                
                if special_chars_data:
                    df_special = pd.DataFrame(special_chars_data)
                    df_special = df_special.sort_values(['Column', 'Count'], ascending=[True, False])
                    df_special.to_excel(writer, sheet_name='Special Characters', index=False)
                else:
                    pd.DataFrame({'Message': ['No special characters detected']}).to_excel(
                        writer, sheet_name='Special Characters', index=False
                    )
            except Exception as e:
                pd.DataFrame({'Error': [f'Could not generate special chars: {str(e)}']}).to_excel(
                    writer, sheet_name='Special Characters', index=False
                )
            
            # 4. Quality Issues (includes duplicate details)
            progress_bar.progress(65, text="Generating Quality Issues...")
            try:
                issues_data = []
                
                # Add columns with issues
                for issue in columns_with_issues:
                    issues_data.append({
                        'Category': 'Column Issue',
                        'Column': issue['column'],
                        'Severity': issue['severity'],
                        'Issue Count': issue['issue_count'],
                        'Reason': issue['reasons']
                    })
                
                # Add duplicate row details
                if dup_details:
                    for dup in dup_details:
                        issues_data.append({
                            'Category': 'Duplicate Row',
                            'Column': 'ALL COLUMNS',
                            'Severity': 'High',
                            'Issue Count': dup['occurrences'],
                            'Reason': f"Exact duplicate: {dup['all_columns_match']} | Sample: {dup['sample'][:100]}"
                        })
                
                # Add other quality issues
                for col_name, prof in state.column_profiles.items():
                    if hasattr(prof, 'business_rule_violations') and prof.business_rule_violations:
                        for viol in prof.business_rule_violations:
                            issues_data.append({
                                'Category': 'Business Rule',
                                'Column': col_name,
                                'Severity': 'High',
                                'Issue Count': 1,
                                'Reason': viol
                            })
                    
                    chars = safe_get_special_chars(prof)
                    if chars:
                        for c in chars:
                            name = c.get('display_name', c.get('name', 'char'))
                            count = c.get('count', 0)
                            issues_data.append({
                                'Category': 'Special Character',
                                'Column': col_name,
                                'Severity': 'Medium',
                                'Issue Count': count,
                                'Reason': f"Contains {name}"
                            })
                
                if issues_data:
                    df_issues = pd.DataFrame(issues_data)
                    # Sort by severity
                    severity_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
                    df_issues['Severity_Rank'] = df_issues['Severity'].map(severity_order)
                    df_issues = df_issues.sort_values('Severity_Rank').drop('Severity_Rank', axis=1)
                    df_issues.to_excel(writer, sheet_name='Quality Issues', index=False)
                else:
                    pd.DataFrame({'Message': ['No quality issues found']}).to_excel(writer, sheet_name='Quality Issues', index=False)
                    
            except Exception as e:
                pd.DataFrame({'Error': [f'Could not generate issues: {str(e)}']}).to_excel(writer, sheet_name='Quality Issues', index=False)

            # 5. Cleansing Plan
            progress_bar.progress(75, text="Generating Cleansing Plan...")
            try:
                cleansing_data = []
                for col_name, prof in state.column_profiles.items():
                    if hasattr(prof, 'cleansing_recommendations') and prof.cleansing_recommendations:
                        for rec in prof.cleansing_recommendations:
                            cleansing_data.append({
                                'Column': col_name, 
                                'Action': rec['action'], 
                                'Description': rec['description'], 
                                'Estimated Impact': rec['impact']
                            })
                
                if cleansing_data:
                    pd.DataFrame(cleansing_data).to_excel(writer, sheet_name='Cleansing Plan', index=False)
                else:
                    pd.DataFrame({'Message': ['No cleansing actions recommended']}).to_excel(writer, sheet_name='Cleansing Plan', index=False)
            except Exception:
                pd.DataFrame({'Error': ['Could not generate cleansing plan']}).to_excel(writer, sheet_name='Cleansing Plan', index=False)
            
            # 6. NEW: Pattern Found Sheet
            progress_bar.progress(90, text="Generating Pattern Found sheet...")
            try:
                pattern_results = detect_patterns_in_dataframe(state.df)
                
                if pattern_results:
                    # Flatten pattern results for Excel
                    pattern_data = []
                    for p in pattern_results:
                        row = {
                            'Column': p['Column'],
                            'Pattern Name': p['Pattern Name'],
                            'Category': p['Category'],
                            'Description': p['Description'],
                            'Regex (Variation-Tolerant)': p['Regex'],
                            'Match Count': p['Match Count'],
                            'Unique Matches': p['Unique Matches'],
                            'Coverage %': p['Coverage %'],
                            'Sample Matches': ', '.join(p['Sample Matches'][:3])
                        }
                        
                        # Add variations if present
                        if p['Variations']:
                            row['Accepted Variations'] = ' | '.join(p['Variations'][:5])
                        else:
                            row['Accepted Variations'] = 'Standard format only'
                        
                        pattern_data.append(row)
                    
                    df_patterns = pd.DataFrame(pattern_data)
                    
                    # Sort by coverage descending
                    df_patterns = df_patterns.sort_values(['Coverage %', 'Match Count'], ascending=[False, False])
                    
                    df_patterns.to_excel(writer, sheet_name='Pattern Found', index=False)
                else:
                    pd.DataFrame({
                        'Message': ['No patterns detected in the dataset'],
                        'Available Patterns': [', '.join(PATTERN_LIBRARY.keys())]
                    }).to_excel(writer, sheet_name='Pattern Found', index=False)
                    
            except Exception as e:
                pd.DataFrame({'Error': [f'Could not generate pattern analysis: {str(e)}']}).to_excel(
                    writer, sheet_name='Pattern Found', index=False
                )
        
        progress_bar.progress(100, text="Finalizing report...")
        time.sleep(0.5)
        progress_bar.empty()
        
        filename = f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.download_button(
            "⬇️ Download Comprehensive Report",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        show_toast("Report generated successfully", "success")
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"Failed to generate report: {str(e)}")


def _generate_json_report():
    """Generate ENHANCED JSON report with pattern detection"""
    import json
    state = st.session_state.app_state
    progress_bar = st.progress(0, text="Initializing JSON report...")
    
    try:
        # Calculate metrics
        overall_quality = _calculate_overall_quality_score(state.column_profiles, len(state.df))
        overall_completeness = _calculate_overall_completeness_score(state.column_profiles)
        total_missing = sum(prof.null_count for prof in state.column_profiles.values())
        dup_count, dup_details = _get_duplicate_analysis(state.df)
        columns_with_issues = _get_columns_with_issues_detailed(state.column_profiles)
        
        # NEW: Pattern detection
        progress_bar.progress(30, text="Detecting patterns...")
        pattern_results = detect_patterns_in_dataframe(state.df)
        
        clean_profiles = {}
        total_cols = len(state.column_profiles)
        
        for idx, (col_name, prof) in enumerate(state.column_profiles.items()):
            if idx % 10 == 0:
                curr_progress = 40 + int((idx / total_cols) * 40)
                progress_bar.progress(curr_progress, text=f"Processing {col_name}...")
            
            try:
                special_chars_list = safe_get_special_chars(prof)
                special_chars_clean = []
                for c in special_chars_list:
                    try:
                        special_chars_clean.append({
                            'symbol': c.get('symbol', ''),
                            'name': c.get('display_name', c.get('name', 'Unknown')),
                            'count': c.get('count', 0),
                            'display': format_char_display(c)
                        })
                    except Exception: continue
                
                # NEW: Add pattern detection per column
                column_patterns = _analyze_column_patterns(state.df, col_name)
                
                clean_profiles[col_name] = {
                    'dtype': prof.dtype,
                    'human_readable_dtype': getattr(prof, 'human_readable_dtype', 'Unknown'),
                    'volume': {
                        'total_rows': prof.total_rows,
                        'non_null_count': getattr(prof, 'non_null_count', 0),
                        'null_count': prof.null_count,
                        'null_percentage': prof.null_percentage
                    },
                    'uniqueness': {
                        'unique_count': prof.unique_count,
                        'unique_percentage': prof.unique_percentage,
                        'duplicate_count': prof.total_rows - prof.unique_count,
                        'duplicate_percentage': getattr(prof, 'duplicate_percentage', 0)
                    },
                    'length_stats': {
                        'min': getattr(prof, 'min_length', 0),
                        'max': getattr(prof, 'max_length', 0),
                        'avg': getattr(prof, 'avg_length', 0),
                        'out_of_bounds': getattr(prof, 'out_of_bounds_count', 0)
                    },
                    'special_characters': special_chars_clean,
                    'special_char_rows': getattr(prof, 'total_special_char_rows', 0),
                    'business_rules': getattr(prof, 'business_rule_violations', []),
                    'risk': {
                        'score': getattr(prof, 'risk_score', 0),
                        'level': getattr(prof, 'risk_level', 'Low'),
                        'issues': getattr(prof, 'key_issues', [])
                    },
                    'cleansing_recommendations': getattr(prof, 'cleansing_recommendations', []),
                    'detected_patterns': column_patterns  # NEW
                }
            except Exception as e:
                clean_profiles[col_name] = {'dtype': getattr(prof, 'dtype', 'unknown'), 'error': f'Could not process: {str(e)}'}
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'filename': getattr(state, 'filename', 'unknown'),
                'total_rows': len(state.df),
                'total_columns': len(state.df.columns),
            },
            'summary': {
                'overall_quality_score': overall_quality,
                'overall_quality_formula': 'Weighted: Completeness(40%) + Uniqueness(30%) + Consistency(20%) + Validity(10%)',
                'overall_completeness_score': overall_completeness,
                'overall_completeness_formula': '(Total Cells - Missing Cells) / Total Cells * 100',
                'missing_cells_count': total_missing,
                'duplicate_rows_count': dup_count,
                'duplicate_details': dup_details,
                'columns_with_issues_count': len(columns_with_issues),
                'columns_with_issues_details': columns_with_issues,
                'patterns_detected_count': len(pattern_results),  # NEW
                'pattern_detection_summary': {  # NEW
                    'total_patterns_found': len(pattern_results),
                    'categories_found': list(set(p['Category'] for p in pattern_results)) if pattern_results else [],
                    'columns_with_patterns': list(set(p['Column'] for p in pattern_results)) if pattern_results else []
                }
            },
            'profiles': clean_profiles,
            'pattern_library': {  # NEW: Include pattern library for reference
                name: {
                    'regex': info['regex'],
                    'description': info['description'],
                    'category': info['category'],
                    'variations': info.get('variations', [])
                }
                for name, info in PATTERN_LIBRARY.items()
            },
            'pattern_matches': pattern_results  # NEW: Detailed pattern matches
        }
        
        json_str = json.dumps(report, indent=2, default=str)
        progress_bar.progress(100, text="Ready!")
        time.sleep(0.5)
        progress_bar.empty()
        
        filename = f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.download_button(
            "⬇️ Download JSON Report",
            data=json_str,
            file_name=filename,
            mime="application/json",
            use_container_width=True
        )
        show_toast("JSON report generated successfully", "success")
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"Failed to generate JSON report: {str(e)}")