"""General data utilities"""

import hashlib
from collections import defaultdict
from typing import List, Optional, Dict
import pandas as pd
import unicodedata
import re
from models import DuplicateGroup


def detect_special_characters(series: pd.Series) -> List[Dict]:
    """Detect special characters in a series using vectorized operations based on the UI's simple requirements"""
    results = []
    
    # Early exit for non-string series
    if not pd.api.types.is_string_dtype(series):
        return []
        
    try:
        # Work with unique values to speed up analysis
        # Value counts gives us frequency of each unique string
        val_counts = series.dropna().value_counts()
        if len(val_counts) == 0:
            return []
            
        unique_series = pd.Series(val_counts.index)
        
        # 1. Check for multiple spaces
        mask = unique_series.str.contains(r'\s{2,}', regex=True)
        if mask.any():
            count = val_counts[unique_series[mask]].sum()
            results.append({
                'key': 'multiple_spaces',
                'symbol': '  ',
                'display_name': 'multiple spaces',
                'count': int(count)
            })
            
        # 2. Check for multiple underscores
        mask = unique_series.str.contains(r'_{2,}', regex=True)
        if mask.any():
            count = val_counts[unique_series[mask]].sum()
            results.append({
                'key': 'multiple_underscores',
                'symbol': '__', 
                'display_name': 'multiple underscores',
                'count': int(count)
            })
            
        # 3. Check for specific individual special characters
        # Common special chars to check
        chars_to_check = [
            ('!', 'exclamation mark'), ('@', 'at symbol'), ('#', 'hash'), 
            ('$', 'dollar'), ('%', 'percent'), ('^', 'caret'), ('&', 'ampersand'),
            ('*', 'asterisk'), ('(', 'left parenthesis'), (')', 'right parenthesis'),
            ('-', 'hyphen/minus'), ('_', 'underscore'), ('+', 'plus'), ('=', 'equals'),
            ('[', 'left bracket'), (']', 'right bracket'), ('{', 'left brace'), 
            ('}', 'right brace'), ('|', 'pipe'), ('\\', 'backslash'), (':', 'colon'),
            (';', 'semicolon'), ('"', 'double quote'), ("'", 'single quote'),
            ('<', 'less than'), ('>', 'greater than'), (',', 'comma'), 
            ('.', 'period'), ('?', 'question mark'), ('/', 'forward slash')
        ]
        
        # Combined regex for all simple chars might be faster but we need individual counts
        # Optimizing: Group check - first check if ANY special char exists
        all_specials = "".join([re.escape(c[0]) for c in chars_to_check])
        if unique_series.str.contains(f"[{all_specials}]", regex=True).any():
            # Only if some exist, check individually
            for char, name in chars_to_check:
                if char in ['_', '-']: # Skip common separators unless many
                    continue
                    
                escaped = re.escape(char)
                mask = unique_series.str.contains(escaped, regex=True)
                if mask.any():
                    count = val_counts[unique_series[mask]].sum()
                    results.append({
                        'key': char,
                        'symbol': char,
                        'display_name': name,
                        'count': int(count)
                    })
        
        # Sort by count
        results.sort(key=lambda x: x['count'], reverse=True)
        
        return results
        
    except Exception:
        return []


def format_special_chars_display(special_chars: List[Dict]) -> str:
    """Format special characters for display"""
    if not special_chars:
        return "No special characters found"
    
    lines = []
    for item in special_chars[:10]:  # Show top 10
        lines.append(f"• **{item['name']}**: {item['count']}x")
        if item['examples']:
            lines.append(f"  _Examples: {item['examples']}_")
    
    return "\n".join(lines)


def find_exact_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> List[DuplicateGroup]:
    """Find exact duplicate rows using vectorized hashing"""
    try:
        if subset is None:
            subset = df.columns.tolist()
        
        # Use pandas hashing which is much faster than row-by-row
        # hash_pandas_object returns a Series of uint64 hashes
        hashes = pd.util.hash_pandas_object(df[subset], index=False)
        
        # Find duplicates based on hash
        # duplicated(keep=False) marks all duplicates as True
        dup_mask = hashes.duplicated(keep=False)
        
        if not dup_mask.any():
            return []
            
        # Get only the duplicate hashes and their indices
        dup_hashes = hashes[dup_mask]
        
        # Group indices by hash
        # This is much faster than iterating
        hash_groups = dup_hashes.groupby(dup_hashes).groups
        
        groups = []
        group_id = 1
        
        for hash_val, indices in hash_groups.items():
            # hash_groups returns indices as Index object, convert to list
            idx_list = indices.tolist()
            
            if len(idx_list) < 2:
                continue
                
            # Double check that they really are identical (hash collisions are rare but possible)
            # Take the first row as reference
            first_row_idx = idx_list[0]
            
            # Since we grouped by hash, we just need to verify values if strictness is needed
            # For performance, we assume hash collision is negligible for 64-bit hash in this context,
            # but to be 100% safe we could do:
            # group_df = df.loc[idx_list, subset]
            # exact_groups = group_df.groupby(list(subset)).groups
            # But that defeats the purpose of hashing speedup if we groupby columns again.
            # Using 64-bit hash from pandas is standard for this.
            
            # Get values for display (convert to dict is slow for big groups, so be careful?)
            # The UI needs `values` as a list of dicts.
            # Limit values stored to avoid memory explosion if group is huge
            
            # Get representative value safely
            try:
                rep_val = str(df.loc[first_row_idx, subset].to_dict())
            except:
                rep_val = "Error getting value"
            
            # For the group values, we must access the main DF
            # Optimization: If group is huge (>100), only store first 100 to save memory in state
            stored_indices = idx_list
            if len(idx_list) > 100:
                stored_values = [df.loc[i, subset].to_dict() for i in idx_list[:100]]
            else:
                stored_values = [df.loc[i, subset].to_dict() for i in idx_list]
            
            groups.append(DuplicateGroup(
                group_id=group_id,
                indices=stored_indices,
                values=stored_values,
                match_type='exact',
                similarity_score=100.0,
                key_columns=subset,
                representative_value=rep_val
            ))
            group_id += 1
            
        return groups
        
    except Exception as e:
        # Fallback to empty if error
        print(f"Error in find_exact_duplicates: {e}")
        return []


def generate_column_suggestions(column, series, null_pct, unique_pct, special_chars, outlier_info, fuzzy_keys, blank_count=None):
    """Generate data quality suggestions for a column"""
    suggestions = []
    if null_pct > 0:
        severity = "🔴" if null_pct > 50 else "🟡" if null_pct > 10 else "🟢"
        suggestions.append(f"{severity} {null_pct:.1f}% missing values")
    
    if blank_count and blank_count.get('empty_string_count', 0) > 0:
        suggestions.append(f"⚠️ {blank_count['empty_string_count']} empty strings (not null)")
    
    if blank_count and blank_count.get('whitespace_only_count', 0) > 0:
        suggestions.append(f"⚠️ {blank_count['whitespace_only_count']} whitespace-only values")
    
    if special_chars:
        suggestions.append(f"🧹 {len(special_chars)} special character types")
    
    if outlier_info["count"] > 0:
        suggestions.append(f"📉 {outlier_info['count']} statistical outliers")
    
    if fuzzy_keys:
        suggestions.append("🔍 Candidate for fuzzy matching")
    
    if unique_pct == 100 and null_pct == 0:
        suggestions.append("✅ Primary key candidate")
    
    if unique_pct < 5:
        suggestions.append("⚠️ Low cardinality - consider categorical")
    
    if pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str)
        if any(s != s.strip() for s in sample):
            suggestions.append("✂️ Leading/trailing whitespace")
        if any(s != s.lower() and s != s.upper() and s != s.title() for s in sample):
            suggestions.append("📝 Inconsistent text casing")
    
    return suggestions