"""Data transformation functions"""

import re
import unicodedata
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from utils.text_processing import AdvancedTitleCase


def update_dataframe(new_df: pd.DataFrame, operation: str):
    """Update dataframe and log operation"""
    from state.session import update_dataframe as session_update_df
    session_update_df(new_df, operation)


def transform_standardize_columns(case: str):
    """Standardize column names"""
    df = st.session_state.df
    original_cols = list(df.columns)
    
    if case == 'snake_case':
        new_cols = [re.sub(r'(?<!^)(?=[A-Z])', '_', str(col)).lower()
                   .replace(' ', '_').replace('-', '_').replace('.', '_').replace('__', '_')
                   for col in df.columns]
    elif case == 'camelCase':
        new_cols = [str(col)[0].lower() + re.sub(r'[_\s](\w)', 
                   lambda m: m.group(1).upper(), str(col)[1:])
                   for col in df.columns]
    elif case == 'PascalCase':
        new_cols = [str(col).replace('_', ' ').replace('-', ' ').title().replace(' ', '')
                   for col in df.columns]
    elif case == 'lower':
        new_cols = [str(col).lower() for col in df.columns]
    elif case == 'upper':
        new_cols = [str(col).upper() for col in df.columns]
    elif case == 'kebab-case':
        new_cols = [str(col).lower().replace(' ', '-').replace('_', '-')
                   for col in df.columns]
    else:
        return
    
    df.columns = new_cols
    update_dataframe(df, f"Standardized columns to {case}")


def transform_remove_exact_duplicates(subset=None, keep='first'):
    """Remove exact duplicates"""
    df = st.session_state.df
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(df)
    update_dataframe(df, f"Removed {removed} exact duplicates")
    return removed


def transform_remove_fuzzy_group(group_indices, strategy='keep_first'):
    """Remove fuzzy duplicate group"""
    df = st.session_state.df
    
    if strategy == 'keep_first':
        keep_idx = group_indices[0]
        drop_indices = group_indices[1:]
    elif strategy == 'keep_last':
        keep_idx = group_indices[-1]
        drop_indices = group_indices[:-1]
    elif strategy == 'merge':
        keep_idx = group_indices[0]
        for idx in group_indices[1:]:
            for col in df.columns:
                if pd.isna(df.loc[keep_idx, col]) and not pd.isna(df.loc[idx, col]):
                    df.loc[keep_idx, col] = df.loc[idx, col]
        drop_indices = group_indices[1:]
    
    df = df.drop(drop_indices)
    update_dataframe(df, f"Merged fuzzy duplicate group ({len(group_indices)} rows)")


def transform_handle_missing(strategy='auto', columns=None):
    """Handle missing values"""
    df = st.session_state.df.copy()
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if df[col].isnull().sum() == 0:
            continue
        
        if strategy == 'auto':
            if pd.api.types.is_numeric_dtype(df[col]):
                skew = df[col].skew()
                fill_value = df[col].median() if abs(skew) > 2 else df[col].mean()
                method = 'median' if abs(skew) > 2 else 'mean'
            else:
                mode_val = df[col].mode()
                fill_value = mode_val[0] if not mode_val.empty else "UNKNOWN"
                method = 'mode'
        elif strategy == 'drop':
            df = df.dropna(subset=[col])
            continue
        elif strategy == 'mean':
            fill_value = df[col].mean()
            method = 'mean'
        elif strategy == 'median':
            fill_value = df[col].median()
            method = 'median'
        elif strategy == 'mode':
            mode_val = df[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else None
            method = 'mode'
        elif strategy == 'constant':
            fill_value = 0 if pd.api.types.is_numeric_dtype(df[col]) else "MISSING"
            method = 'constant'
        else:
            continue
        
        df[col] = df[col].fillna(fill_value)
    
    update_dataframe(df, f"Handled missing values using {strategy}")


def transform_clean_special_chars(columns=None):
    """Clean special characters"""
    df = st.session_state.df.copy()
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        
        df[col] = df[col].astype(str).replace('nan', np.nan).replace('None', np.nan)
        df[col] = df[col].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', str(x))
                                if unicodedata.category(c) != 'Mn') if pd.notna(x) else x)
        df[col] = df[col].apply(lambda x: ' '.join(str(x).split()) if pd.notna(x) else x)
        df[col] = df[col].apply(lambda x: re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', str(x))
                                if pd.notna(x) else x)
        
        replacements = {'\xa0': ' ', '\u200b': '', '\ufeff': '', '"': '"', '"': '"', ''': "'", ''': "'"}
        for old, new in replacements.items():
            df[col] = df[col].str.replace(old, new, regex=False)
    
    update_dataframe(df, f"Cleaned special characters in {len(columns)} columns")


def transform_standardize_text(columns, case='lower', title_style='apa'):
    """Standardize text case"""
    df = st.session_state.df.copy()
    
    for col in columns:
        if case == 'lower':
            df[col] = df[col].str.lower()
        elif case == 'upper':
            df[col] = df[col].str.upper()
        elif case == 'title':
            df[col] = df[col].apply(lambda x: AdvancedTitleCase.convert(x, style=title_style) if pd.notna(x) else x)
        elif case == 'sentence':
            df[col] = df[col].apply(lambda x: AdvancedTitleCase.convert(x, style='sentence') if pd.notna(x) else x)
        elif case == 'capitalize':
            df[col] = df[col].str.capitalize()
    
    update_dataframe(df, f"Standardized text case to {case}")


def transform_trim_whitespace(columns=None):
    """Trim whitespace"""
    df = st.session_state.df.copy()
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    
    update_dataframe(df, "Trimmed whitespace")


def transform_convert_types(conversions: Dict[str, str]):
    """Convert data types"""
    df = st.session_state.df.copy()
    
    for col, dtype in conversions.items():
        if col not in df.columns:
            continue
        
        try:
            if dtype == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif dtype == 'category':
                df[col] = df[col].astype('category')
            elif dtype == 'int':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif dtype == 'float':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif dtype == 'str':
                df[col] = df[col].astype(str).replace('nan', np.nan)
            elif dtype == 'bool':
                df[col] = df[col].astype(bool)
        except Exception as e:
            st.error(f"Error converting {col}: {e}")
    
    update_dataframe(df, f"Converted types for {len(conversions)} columns")


def transform_remove_outliers(columns=None, method='iqr'):
    """Remove outliers"""
    df = st.session_state.df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    before = len(df)
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper) | df[col].isnull()]
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < 3]
    
    removed = before - len(df)
    update_dataframe(df, f"Removed {removed} outliers using {method}")
    return removed


def transform_apply_regex(column: str, pattern: str, replacement: str = '', action: str = 'replace'):
    """Apply regex transformation to a column"""
    df = st.session_state.df.copy()
    
    if column not in df.columns:
        st.error(f"Column {column} not found")
        return
    
    try:
        if action == 'replace':
            df[column] = df[column].astype(str).replace('nan', np.nan).replace('None', np.nan)
            df[column] = df[column].apply(lambda x: re.sub(pattern, replacement, str(x)) if pd.notna(x) else x)
        elif action == 'extract':
            extracted = df[column].astype(str).str.extract(pattern)
            if not extracted.empty:
                new_col_name = f"{column}_extracted"
                df[new_col_name] = extracted[0]
        elif action == 'match':
            mask = df[column].astype(str).str.match(pattern, na=False)
            df = df[mask]
        elif action == 'filter':
            mask = df[column].astype(str).str.contains(pattern, na=False, regex=True)
            df = df[mask]
        
        update_dataframe(df, f"Applied regex '{pattern}' to {column}")
    except re.error as e:
        st.error(f"Invalid regex pattern: {e}")
    except Exception as e:
        st.error(f"Error applying regex: {e}")


def transform_apply_business_rule(column: str, rule_type: str, **kwargs):
    """Apply business rules to validate/transform data"""
    df = st.session_state.df.copy()
    
    if column not in df.columns:
        st.error(f"Column {column} not found")
        return
    
    try:
        if rule_type == 'range':
            min_val = kwargs.get('min_val')
            max_val = kwargs.get('max_val')
            if min_val is not None:
                df = df[df[column] >= min_val]
            if max_val is not None:
                df = df[df[column] <= max_val]
        
        elif rule_type == 'length':
            min_len = kwargs.get('min_len', 0)
            max_len = kwargs.get('max_len', float('inf'))
            mask = df[column].astype(str).str.len().between(min_len, max_len)
            df = df[mask]
        
        elif rule_type == 'email_format':
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            mask = df[column].astype(str).str.match(email_pattern, na=False)
            invalid_count = (~mask).sum()
            df = df[mask]
            st.info(f"Removed {invalid_count} invalid email addresses")
        
        elif rule_type == 'phone_format':
            # Remove non-numeric and check length
            cleaned = df[column].astype(str).str.replace(r'\D', '', regex=True)
            mask = cleaned.str.len().between(10, 15)
            invalid_count = (~mask).sum()
            df = df[mask]
            st.info(f"Removed {invalid_count} invalid phone numbers")
        
        elif rule_type == 'date_range':
            min_date = kwargs.get('min_date')
            max_date = kwargs.get('max_date')
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                if min_date:
                    df = df[df[column] >= pd.to_datetime(min_date)]
                if max_date:
                    df = df[df[column] <= pd.to_datetime(max_date)]
        
        elif rule_type == 'allowed_values':
            allowed = kwargs.get('allowed_values', [])
            if allowed:
                mask = df[column].isin(allowed)
                invalid_count = (~mask).sum()
                df = df[mask]
                st.info(f"Removed {invalid_count} values not in allowed list")
        
        elif rule_type == 'pattern':
            pattern = kwargs.get('pattern', '')
            mask = df[column].astype(str).str.match(pattern, na=False)
            invalid_count = (~mask).sum()
            df = df[mask]
            st.info(f"Removed {invalid_count} values not matching pattern")
        
        elif rule_type == 'not_null':
            df = df[df[column].notna() & (df[column] != '')]
        
        elif rule_type == 'unique':
            df = df.drop_duplicates(subset=[column], keep='first')
        
        update_dataframe(df, f"Applied business rule '{rule_type}' to {column}")
        
    except Exception as e:
        st.error(f"Error applying business rule: {e}")


def transform_auto_fix():
    """Apply all automatic fixes"""
    df = st.session_state.df.copy()
    
    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(keep='first')
    dup_removed = before - len(df)
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median() if abs(df[col].skew()) > 2 else df[col].mean()
        else:
            mode_val = df[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else "UNKNOWN"
        df[col] = df[col].fillna(fill_value)
    
    # Clean special characters
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace('nan', np.nan)
        df[col] = df[col].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', str(x))
                                if unicodedata.category(c) != 'Mn') if pd.notna(x) else x)
        df[col] = df[col].str.replace('\xa0', ' ', regex=False)
    
    # Trim whitespace
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    
    # Standardize column names
    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', str(col)).lower()
                 .replace(' ', '_').replace('-', '_').replace('.', '_')
                 for col in df.columns]
    
    update_dataframe(df, f"Auto-fix: removed {dup_removed} duplicates, cleaned text, standardized names")