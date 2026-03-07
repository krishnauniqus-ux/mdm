"""Utilities for Data Quality rule detection"""

import pandas as pd
import re
from typing import Dict, Optional

def extract_existing_rules(df: pd.DataFrame) -> Dict[str, str]:
    """
    Extract existing DQ rules from the dataframe.
    Looks for a column named 'DQ Rules' or similar.
    """
    rules = {}
    
    # 1. Look for a specific column for rules
    rule_col_patterns = ['DQ Rules', 'Validation Rules', 'Rules', 'Quality Rules']
    rule_col = None
    
    for col in df.columns:
        if any(pattern.lower() in str(col).lower() for pattern in rule_col_patterns):
            rule_col = col
            break
            
    if rule_col:
        # Assuming the row index or first few columns match other data columns
        # This is a heuristic. Let's try to map rules to columns.
        # Often a "DQ Rules" column might have values like "ColumnA: Must not be null"
        for val in df[rule_col].dropna().unique():
            val_str = str(val)
            # Try to find ":" or "-" as separator
            if ':' in val_str:
                parts = val_str.split(':', 1)
                col_name = parts[0].strip()
                rule_stmt = parts[1].strip()
                if col_name in df.columns:
                    rules[col_name] = rule_stmt
            elif '-' in val_str:
                parts = val_str.split('-', 1)
                col_name = parts[0].strip()
                rule_stmt = parts[1].strip()
                if col_name in df.columns:
                    rules[col_name] = rule_stmt
                    
    # 2. Look for rules within column names themselves (e.g. "Amount (char 100)")
    for col in df.columns:
        if col not in rules:
            col_str = str(col)
            if 'char' in col_str.lower():
                match = re.search(r'char\s*(\d+)', col_str, re.IGNORECASE)
                if match:
                    length = match.group(1)
                    rules[col] = f"Maximum length = {length} characters"
                    
    return rules

def get_metadata_rows(file_path: str, sheet_name: Optional[str] = None, header_row: int = 0) -> pd.DataFrame:
    """Load rows from 0 to header_row-1 as potential rule/metadata rows"""
    if header_row <= 0:
        return pd.DataFrame()
        
    file_ext = file_path.split('.')[-1].lower()
    if file_ext in ['xlsx', 'xls']:
        # Load first few rows without header
        return pd.read_excel(file_path, sheet_name=sheet_name, nrows=header_row, header=None)
    elif file_ext in ['csv']:
        return pd.read_csv(file_path, nrows=header_row, header=None)
    return pd.DataFrame()
