"""Data Profiling Component - Final Enhanced Version with Universal Validations"""

import streamlit as st
import pandas as pd
import io
import re
import unicodedata
from datetime import datetime
from collections import Counter, defaultdict
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from state.session import st, show_toast
from core.profiler import DataProfilerEngine

# ==========================================
# UNIVERSAL VALIDATION PATTERNS LIBRARY
# ==========================================

VALIDATION_PATTERNS = {
    # Indian Government IDs
    'Aadhaar': {
        'regex': r'^(?!.*([0-9])\1{3})[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}$',
        'description': '12-digit Indian Aadhaar (with optional spaces)',
        'category': 'Government ID',
        'country': 'India',
        'format': 'XXXX XXXX XXXX or XXXXXXXXXXXX',
        'example': '1234 5678 9012'
    },
    'PAN': {
        'regex': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$',
        'description': 'Indian PAN (Permanent Account Number)',
        'category': 'Government ID',
        'country': 'India',
        'format': 'AAAAA9999A',
        'example': 'ABCDE1234F'
    },
    'GSTIN': {
        'regex': r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$',
        'description': 'Indian GST Identification Number',
        'category': 'Government ID',
        'country': 'India',
        'format': '22AAAAA0000A1Z5',
        'example': '22ABCDE1234F1Z5'
    },
    'Voter ID': {
        'regex': r'^[A-Z]{3}[0-9]{7}$',
        'description': 'Indian Voter ID (EPIC)',
        'category': 'Government ID',
        'country': 'India',
        'format': 'AAA1234567',
        'example': 'ABC1234567'
    },
    'Driving License': {
        'regex': r'^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$',
        'description': 'Indian Driving License',
        'category': 'Government ID',
        'country': 'India',
        'format': 'MH0212345678901',
        'example': 'MH0212345678901'
    },

    # Global Government IDs
    'Passport': {
        'regex': r'^[A-Z]{1}[0-9]{7}$',
        'description': 'International Passport (most countries)',
        'category': 'Government ID',
        'country': 'Global',
        'format': 'A1234567',
        'example': 'A1234567'
    },
    'SSN': {
        'regex': r'^(?!666|000|9\d{2})\d{3}-?(?!00)\d{2}-?(?!0{4})\d{4}$',
        'description': 'US Social Security Number',
        'category': 'Government ID',
        'country': 'USA',
        'format': 'XXX-XX-XXXX',
        'example': '123-45-6789'
    },
    'National Insurance': {
        'regex': r'^[A-Z]{2}[0-9]{6}[A-Z]{1}$',
        'description': 'UK National Insurance Number',
        'category': 'Government ID',
        'country': 'UK',
        'format': 'AA123456A',
        'example': 'AB123456C'
    },

    # Contact Information
    'Mobile India': {
        'regex': r'^(\+91[-\s]?)?[6-9]\d{9}$',
        'description': 'Indian Mobile Number (10 digits)',
        'category': 'Contact',
        'country': 'India',
        'format': '+91XXXXXXXXXX or 9XXXXXXXXX',
        'example': '+919876543210'
    },
    'Mobile US': {
        'regex': r'^(\+1[-\s]?)?\(?[0-9]{3}\)?[-\s]?[0-9]{3}[-\s]?[0-9]{4}$',
        'description': 'US Mobile/Phone Number',
        'category': 'Contact',
        'country': 'USA',
        'format': '+1 (XXX) XXX-XXXX',
        'example': '+1 (555) 123-4567'
    },
    'Mobile UK': {
        'regex': r'^(\+44[-\s]?)?07[0-9]{9}$',
        'description': 'UK Mobile Number',
        'category': 'Contact',
        'country': 'UK',
        'format': '+447XXXXXXXXX',
        'example': '+447123456789'
    },
    'Mobile Global': {
        'regex': r'^\+?[1-9]\d{7,14}$',
        'description': 'Generic International Mobile',
        'category': 'Contact',
        'country': 'Global',
        'format': '+XXXXXXXXXXX',
        'example': '+1234567890'
    },

    # Email & Web
    'Email': {
        'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'description': 'Standard Email Address',
        'category': 'Contact',
        'country': 'Global',
        'format': 'name@domain.com',
        'example': 'user@example.com'
    },
    'URL': {
        'regex': r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$',
        'description': 'HTTP/HTTPS URL',
        'category': 'Web',
        'country': 'Global',
        'format': 'https://domain.com',
        'example': 'https://www.example.com'
    },

    # Financial
    'Credit Card': {
        'regex': r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})$',
        'description': 'Credit Card (Visa, MasterCard, AmEx, etc.)',
        'category': 'Financial',
        'country': 'Global',
        'format': 'XXXXXXXXXXXXXXXX',
        'example': '4111111111111111'
    },
    'IBAN': {
        'regex': r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}(?:[A-Z0-9]?){0,16}$',
        'description': 'International Bank Account Number',
        'category': 'Financial',
        'country': 'Global',
        'format': 'XX00XXXX00000000000000',
        'example': 'GB82WEST12345698765432'
    },
    'SWIFT': {
        'regex': r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$',
        'description': 'SWIFT/BIC Code',
        'category': 'Financial',
        'country': 'Global',
        'format': 'AAAA BB CC (XXX)',
        'example': 'CHASUS33'
    },
    'IFSC': {
        'regex': r'^[A-Z]{4}0[A-Z0-9]{6}$',
        'description': 'Indian Financial System Code',
        'category': 'Financial',
        'country': 'India',
        'format': 'AAAA0XXXXXX',
        'example': 'HDFC0001234'
    },
    'UPI': {
        'regex': r'^[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z][a-zA-Z]{2,64}$',
        'description': 'UPI ID (India)',
        'category': 'Financial',
        'country': 'India',
        'format': 'name@bank',
        'example': 'user@okaxis'
    },

    # Address & Location
    'PIN Code India': {
        'regex': r'^[1-9][0-9]{5}$',
        'description': 'Indian PIN/ZIP Code',
        'category': 'Address',
        'country': 'India',
        'format': 'XXXXXX',
        'example': '110001'
    },
    'ZIP US': {
        'regex': r'^[0-9]{5}(?:-[0-9]{4})?$',
        'description': 'US ZIP Code',
        'category': 'Address',
        'country': 'USA',
        'format': 'XXXXX or XXXXX-XXXX',
        'example': '12345 or 12345-6789'
    },
    'Postal UK': {
        'regex': r'^[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}$',
        'description': 'UK Postal Code',
        'category': 'Address',
        'country': 'UK',
        'format': 'AA9A 9AA',
        'example': 'SW1A 1AA'
    },

    # Technical
    'IPv4': {
        'regex': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        'description': 'IPv4 Address',
        'category': 'Technical',
        'country': 'Global',
        'format': 'XXX.XXX.XXX.XXX',
        'example': '192.168.1.1'
    },
    'IPv6': {
        'regex': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
        'description': 'IPv6 Address',
        'category': 'Technical',
        'country': 'Global',
        'format': 'XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX',
        'example': '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    },
    'MAC Address': {
        'regex': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
        'description': 'MAC Address',
        'category': 'Technical',
        'country': 'Global',
        'format': 'XX:XX:XX:XX:XX:XX',
        'example': '00:1B:44:11:3A:B7'
    },

    # Dates
    'Date ISO': {
        'regex': r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$',
        'description': 'ISO Date (YYYY-MM-DD)',
        'category': 'Date',
        'country': 'Global',
        'format': 'YYYY-MM-DD',
        'example': '2024-01-15'
    },
    'Date US': {
        'regex': r'^(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}$',
        'description': 'US Date (MM/DD/YYYY)',
        'category': 'Date',
        'country': 'USA',
        'format': 'MM/DD/YYYY',
        'example': '01/15/2024'
    },
    'DateTime ISO': {
        'regex': r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])[T ](0[0-9]|1[0-9]|2[0-3]):[0-5]\d:[0-5]\d$',
        'description': 'ISO DateTime',
        'category': 'Date',
        'country': 'Global',
        'format': 'YYYY-MM-DD HH:MM:SS',
        'example': '2024-01-15 14:30:00'
    }
}

# ==========================================
# CORE FUNCTIONS
# ==========================================

def detect_all_validations(df):
    """Detect all validation patterns in dataframe"""
    results = []

    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().astype(str).head(2000).tolist()
        if not sample:
            continue

        for val_name, val_info in VALIDATION_PATTERNS.items():
            valid_count = 0
            invalid_count = 0
            samples = {'valid': [], 'invalid': []}

            for val in sample[:500]:  # Check first 500 for performance
                if re.match(val_info['regex'], val.strip()):
                    valid_count += 1
                    if len(samples['valid']) < 3:
                        samples['valid'].append(val)
                else:
                    # Check if it looks like it should match (contains similar chars)
                    if _looks_like_pattern(val, val_name):
                        invalid_count += 1
                        if len(samples['invalid']) < 3:
                            samples['invalid'].append(val)

            coverage = (valid_count / len(sample[:500])) * 100 if sample else 0

            if coverage > 5 or valid_count > 10:  # Threshold for detection
                results.append({
                    'Column': col,
                    'Validation Type': val_name,
                    'Category': val_info['category'],
                    'Country': val_info['country'],
                    'Valid Count': valid_count,
                    'Invalid Count': invalid_count,
                    'Coverage %': round(coverage, 2),
                    'Format': val_info['format'],
                    'Example': val_info['example'],
                    'Valid Samples': samples['valid'],
                    'Invalid Samples': samples['invalid']
                })

    return results


def _looks_like_pattern(value, pattern_name):
    """Check if value looks like it should match the pattern"""
    val = value.strip()
    if not val:
        return False

    checks = {
        'Aadhaar': lambda v: len(re.sub(r'\D', '', v)) == 12,
        'PAN': lambda v: len(v) == 10 and v[:5].isalpha() and v[5:9].isdigit(),
        'Mobile India': lambda v: len(re.sub(r'\D', '', v)) >= 10,
        'Email': lambda v: '@' in v and '.' in v.split('@')[-1],
        'Credit Card': lambda v: len(re.sub(r'\D', '', v)) >= 13,
        'Passport': lambda v: len(v) == 8 and v[0].isalpha(),
        'GSTIN': lambda v: len(v) == 15,
        'Date ISO': lambda v: len(v) == 10 and v[4] == '-' and v[7] == '-',
        'URL': lambda v: v.startswith('http'),
        'IPv4': lambda v: len(v.split('.')) == 4
    }

    return checks.get(pattern_name, lambda v: False)(val)


def analyze_column_validations(df, col):
    """Analyze validations for specific column"""
    sample = df[col].dropna().astype(str).head(1000).tolist()
    if not sample:
        return []

    results = []
    for val_name, val_info in VALIDATION_PATTERNS.items():
        valid = sum(1 for v in sample if re.match(val_info['regex'], v.strip()))
        if valid > 0:
            coverage = (valid / len(sample)) * 100
            results.append({
                'type': val_name,
                'category': val_info['category'],
                'coverage': round(coverage, 1),
                'valid_count': valid,
                'invalid_count': len(sample) - valid,
                'format': val_info['format']
            })

    return sorted(results, key=lambda x: x['coverage'], reverse=True)


def find_duplicate_groups(df, col):
    """Find duplicate groups in a column"""
    if col not in df.columns:
        return []

    value_counts = df[col].value_counts()
    duplicates = value_counts[value_counts > 1]

    groups = []
    for val, count in duplicates.head(10).items():
        # Find all rows with this value
        matching_rows = df[df[col] == val]
        groups.append({
            'value': str(val)[:50],
            'count': int(count),
            'percentage': round((count / len(df)) * 100, 2),
            'row_indices': matching_rows.index.tolist()[:5]  # First 5 occurrences
        })

    return groups


def generate_match_rules(df, profiles):
    """Generate match rules - FIXED: excludes 100% unique columns from exact match"""
    rules = []
    counter = 1

    # Analyze columns
    analysis = {}
    for col, prof in profiles.items():
        dup_count = prof.total_rows - prof.unique_count
        dup_pct = (dup_count / prof.total_rows * 100) if prof.total_rows > 0 else 0

        analysis[col] = {
            'null_pct': prof.null_percentage,
            'unique_pct': prof.unique_percentage,
            'dup_pct': dup_pct,
            'dup_count': dup_count,
            'is_text': prof.dtype == 'object',
            'is_num': any(t in prof.dtype for t in ['int', 'float']),
            'avg_len': getattr(prof, 'avg_length', 0),
            'max_len': getattr(prof, 'max_length', 0),
            'min_len': getattr(prof, 'min_length', 0),
            'validations': analyze_column_validations(df, col),
            'total_rows': prof.total_rows
        }

    # EXACT MATCH CANDIDATES - Must have duplicates (not 100% unique)
    exact_candidates = []
    for col, a in analysis.items():
        # SKIP if 100% unique - cannot be exact match key
        if a['unique_pct'] == 100 or a['dup_count'] == 0:
            continue

        score = 0
        reasons = []

        # High uniqueness but NOT 100% (95-99.9%)
        if 95 <= a['unique_pct'] < 100:
            score += 35
            reasons.append(f"Near-unique ({a['unique_pct']:.1f}%)")
        elif 80 <= a['unique_pct'] < 95:
            score += 25
            reasons.append(f"High uniqueness ({a['unique_pct']:.1f}%)")

        # Low null rate
        if a['null_pct'] < 1:
            score += 20
            reasons.append("Complete data (no nulls)")
        elif a['null_pct'] < 5:
            score += 15
            reasons.append("Low null rate")

        # Fixed length (good for codes/IDs)
        if a['is_text'] and a['max_len'] == a['min_len'] and 4 <= a['avg_len'] <= 20:
            score += 25
            reasons.append(f"Fixed length ({int(a['avg_len'])} chars)")

        # Has validation pattern (Government ID, etc.)
        for v in a['validations']:
            if v['category'] == 'Government ID' and v['coverage'] > 50:
                score += 30
                reasons.append(f"ID format: {v['type']}")
                break

        # Low duplicate percentage (good quality)
        if 0 < a['dup_pct'] < 5:
            score += 15
            reasons.append(f"Low duplicates ({a['dup_pct']:.1f}%)")

        if score >= 50 and a['dup_count'] > 0:
            exact_candidates.append({
                'column': col,
                'score': score,
                'reasons': reasons,
                'dup_count': a['dup_count'],
                'unique_pct': a['unique_pct']
            })

    exact_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Generate Exact Match Rules (max 4)
    for cand in exact_candidates[:4]:
        prob = "Strongest" if cand['score'] >= 85 else "Very Strong" if cand['score'] >= 75 else "Strong" if cand['score'] >= 65 else "Good"
        rules.append({
            'Rule No': f"R{counter:02d}",
            'Rule Type': 'Exact',
            'Columns': cand['column'],
            'Match Probability': prob,
            'Rationale': f"{'; '.join(cand['reasons'][:2])} | Duplicates: {cand['dup_count']}",
            'Confidence': cand['score']
        })
        counter += 1

    # FUZZY MATCH CANDIDATES - Text fields, medium uniqueness
    fuzzy_candidates = []
    for col, a in analysis.items():
        if not a['is_text']:
            continue

        score = 0
        reasons = []

        # Medium uniqueness (30-90%) - sweet spot for fuzzy
        if 30 <= a['unique_pct'] <= 90:
            score += 35
            reasons.append(f"Medium uniqueness ({a['unique_pct']:.1f}%)")
        elif 10 <= a['unique_pct'] < 30:
            score += 20
            reasons.append(f"Low-medium uniqueness ({a['unique_pct']:.1f}%)")

        # Text length suitable for names/descriptions
        if 10 <= a['avg_len'] <= 100:
            score += 25
            reasons.append(f"Name/description length ({a['avg_len']:.0f} chars)")
        elif a['avg_len'] > 100:
            score += 15
            reasons.append("Long text field")

        # Name indicators in column name
        name_indicators = ['name', 'desc', 'title', 'product', 'customer', 'company', 'vendor', 'supplier', 'brand', 'item']
        if any(ind in col.lower() for ind in name_indicators):
            score += 25
            reasons.append("Name/description column")

        # Has some duplicates (can be matched)
        if a['dup_count'] > 1:
            score += 15
            reasons.append(f"Has duplicates to match ({a['dup_count']})")

        if score >= 45:
            fuzzy_candidates.append({
                'column': col,
                'score': score,
                'reasons': reasons,
                'unique_pct': a['unique_pct']
            })

    fuzzy_candidates.sort(key=lambda x: x['score'], reverse=True)

    # Generate Fuzzy Match Rules (max 4)
    for cand in fuzzy_candidates[:4]:
        prob = "Strong" if cand['score'] >= 70 else "Good" if cand['score'] >= 60 else "Medium"
        rules.append({
            'Rule No': f"R{counter:02d}",
            'Rule Type': 'Fuzzy',
            'Columns': cand['column'],
            'Match Probability': prob,
            'Rationale': '; '.join(cand['reasons'][:3]),
            'Confidence': cand['score']
        })
        counter += 1

    # COMBINED RULES - Pair Exact + Fuzzy
    if exact_candidates and fuzzy_candidates:
        for e in exact_candidates[:2]:
            for f in fuzzy_candidates[:3]:
                if e['column'] != f['column'] and len(rules) < 10:
                    score = (e['score'] + f['score']) / 2
                    prob = "Enterprise" if score >= 75 else "Fashion Strong" if score >= 65 else "Better"
                    rules.append({
                        'Rule No': f"R{counter:02d}",
                        'Rule Type': 'Combined',
                        'Columns': f"{e['column']} (Exact) + {f['column']} (Fuzzy)",
                        'Match Probability': prob,
                        'Rationale': f"Exact on {e['column']}; Fuzzy match on {f['column']}",
                        'Confidence': score
                    })
                    counter += 1
                    if len(rules) >= 10:
                        break
            if len(rules) >= 10:
                break

    # Ensure minimum 1 rule, maximum 10
    if not rules and analysis:
        first = list(analysis.keys())[0]
        rules.append({
            'Rule No': 'R01',
            'Rule Type': 'Exact',
            'Columns': first,
            'Match Probability': 'Medium',
            'Rationale': 'Default fallback rule',
            'Confidence': 50
        })

    # Sort by confidence and renumber
    rules.sort(key=lambda x: x['Confidence'], reverse=True)
    for i, r in enumerate(rules[:10], 1):
        r['Rule No'] = f"R{i:02d}"

    return rules[:10]


def generate_dq_rules(df, profiles):
    """Generate comprehensive DQ rules with universal validations"""
    rules = []

    for col, prof in profiles.items():
        a = {
            'null_pct': prof.null_percentage,
            'null_count': prof.null_count,
            'unique_pct': prof.unique_percentage,
            'dup_count': prof.total_rows - prof.unique_count,
            'total': prof.total_rows,
            'is_text': prof.dtype == 'object',
            'is_num': any(t in prof.dtype for t in ['int', 'float']),
            'is_date': 'date' in prof.dtype,
            'min_len': getattr(prof, 'min_length', 0),
            'max_len': getattr(prof, 'max_length', 0),
            'oob': getattr(prof, 'out_of_bounds_count', 0),
            'validations': analyze_column_validations(df, col),
            'special': safe_get_special_chars(prof)
        }

        # 1. COMPLETENESS
        if a['null_pct'] > 0:
            sev = "CRITICAL" if a['null_pct'] > 50 else "HIGH" if a['null_pct'] > 20 else "MEDIUM" if a['null_pct'] > 5 else "LOW"
            rules.append({
                'Column': col, 'Dimension': 'Completeness', 'Type': 'NOT_NULL',
                'Condition': f'{col} IS NOT NULL', 'Severity': sev,
                'Action': 'Route to Data Steward' if sev in ['CRITICAL','HIGH'] else 'Flag for review',
                'Found': f"{a['null_count']} nulls ({a['null_pct']:.1f}%)",
                'Expected': 'Complete data'
            })

        # 2. UNIQUENESS - Only flag if near-unique (potential key)
        if 95 <= a['unique_pct'] < 100 and a['dup_count'] > 0:
            rules.append({
                'Column': col, 'Dimension': 'Uniqueness', 'Type': 'UNIQUE',
                'Condition': f'{col} must be unique', 'Severity': 'HIGH',
                'Action': 'Escalate to Master Data team',
                'Found': f"{a['dup_count']} duplicates ({100-a['unique_pct']:.2f}%)",
                'Expected': '100% unique (identifier)'
            })

        # 3. VALIDATION RULES - Based on detected patterns
        for v in a['validations']:
            if v['coverage'] < 100:
                non_match = 100 - v['coverage']
                sev = "HIGH" if non_match > 20 else "MEDIUM" if non_match > 5 else "LOW"

                rules.append({
                    'Column': col,
                    'Dimension': 'Validity',
                    'Type': f"FORMAT_CHECK ({v['type']})",
                    'Condition': f"Matches {v['type']} format: {v['format']}",
                    'Severity': sev,
                    'Action': 'Auto-standardize' if sev == 'LOW' else 'Trigger validation workflow',
                    'Found': f"{v['invalid_count']} invalid ({non_match:.1f}%)",
                    'Expected': f"Valid {v['type']} format"
                })

        # 4. LENGTH VALIDATION
        if a['is_text'] and a['max_len'] > 0:
            variance = a['max_len'] - a['min_len']
            if variance > 20:
                rules.append({
                    'Column': col, 'Dimension': 'Validity', 'Type': 'LENGTH_CHECK',
                    'Condition': f'Length {a["min_len"]}-{a["max_len"]}',
                    'Severity': 'MEDIUM',
                    'Action': 'Notify business team',
                    'Found': f'High variance ({variance} chars)',
                    'Expected': 'Consistent length'
                })

        # 5. ACCURACY - Out of bounds
        if a['oob'] > 0:
            rules.append({
                'Column': col, 'Dimension': 'Accuracy', 'Type': 'RANGE_CHECK',
                'Condition': 'Within expected bounds', 'Severity': 'HIGH',
                'Action': 'Block publish to ERP',
                'Found': f"{a['oob']} out of bounds",
                'Expected': 'Within defined limits'
            })

        # 6. CONSISTENCY - Special characters
        if a['special']:
            chars = [c.get('symbol','') for c in a['special'][:3]]
            rules.append({
                'Column': col, 'Dimension': 'Consistency', 'Type': 'SPECIAL_CHARS',
                'Condition': f'Clean text (no {chars})', 'Severity': 'MEDIUM',
                'Action': 'Auto-standardize',
                'Found': f"{len(a['special'])} special char types",
                'Expected': 'Alphanumeric only'
            })

        # 7. ENUMERATION CHECK (low cardinality)
        if a['is_text'] and prof.unique_count <= 15 and a['total'] > 50:
            vals = df[col].dropna().unique()[:10].tolist()
            rules.append({
                'Column': col, 'Dimension': 'Validity', 'Type': 'ENUM_CHECK',
                'Condition': 'Valid enumerated values', 'Severity': 'MEDIUM',
                'Action': 'MASTER_LOOKUP',
                'Found': f"{prof.unique_count} distinct values",
                'Expected': f'From approved list: {vals[:5]}'
            })

        # 8. DATE VALIDATION
        if a['is_date']:
            rules.append({
                'Column': col, 'Dimension': 'Timeliness', 'Type': 'DATE_RANGE',
                'Condition': 'Not in future', 'Severity': 'HIGH',
                'Action': 'Prevent activation',
                'Found': 'Check for future dates',
                'Expected': 'Historical/current dates'
            })

    # Sort by severity
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    rules.sort(key=lambda x: severity_order.get(x['Severity'], 4))

    return rules


def safe_get_special_chars(prof):
    try:
        if hasattr(prof, 'special_chars') and prof.special_chars:
            return [c for c in prof.special_chars if isinstance(c, dict) and 'count' in c]
    except: pass
    return []


def _remove_ghost_columns():
    state = st.session_state.app_state
    if state.df is None: return False
    ghosts = [c for c in state.df.columns if str(c).startswith('Unnamed:') and state.df[c].isnull().all()]
    if ghosts:
        state.df = state.df.drop(columns=ghosts)
        for c in ghosts: state.column_profiles.pop(c, None)
        show_toast(f"Removed {len(ghosts)} empty columns", "success")
        return True
    return False

# ==========================================
# ENHANCED UI WITH VISUALIZATIONS
# ==========================================

def render_data_profiling():
    state = st.session_state.app_state
    if state.df is None:
        st.info("📤 Please load data first")
        return

    _remove_ghost_columns()

    st.set_page_config(page_title="Data Profiler", layout="wide")
    st.title("🔍 Enterprise Data Profiler")

    # Executive Dashboard
    _render_executive_dashboard()

    # Main Tabs
    tabs = st.tabs(["📊 Overview", "📋 Column Profiles", "✅ Validations", "🎯 Match Rules", "⚠️ Quality Issues", "📥 Export"])

    with tabs[0]: _render_overview_tab()
    with tabs[1]: _render_profiles_tab()
    with tabs[2]: _render_validations_tab()
    with tabs[3]: _render_match_rules_tab()
    with tabs[4]: _render_issues_tab()
    with tabs[5]: _render_export_tab()


def _render_executive_dashboard():
    state = st.session_state.app_state
    df = state.df
    profiles = state.column_profiles

    # Calculate metrics
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    missing_cells = sum(p.null_count for p in profiles.values())
    completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells else 0

    # Quality score
    quality_scores = []
    for p in profiles.values():
        comp = getattr(p, 'non_null_percentage', 100)
        uniq = min(100, p.unique_percentage * 1.2) if p.unique_percentage < 100 else 90
        consistency = 100 - (20 if hasattr(p, 'formatting_info') and p.formatting_info and not p.formatting_info.get('consistent_case', True) else 0)
        validity = 100 - min(20, len(safe_get_special_chars(p)) * 2)
        quality_scores.append(comp * 0.4 + uniq * 0.3 + consistency * 0.2 + validity * 0.1)
    avg_quality = round(sum(quality_scores)/len(quality_scores), 1) if quality_scores else 0

    # KPI Row
    cols = st.columns(6)
    kpi_data = [
        ("📊", "Rows", f"{total_rows:,}"),
        ("📁", "Columns", total_cols),
        ("⭐", "Quality", f"{avg_quality:.0f}%"),
        ("✅", "Completeness", f"{completeness:.1f}%"),
        ("⚠️", "Missing", f"{missing_cells:,}"),
        ("🎯", "Fill Rate", f"{100-(missing_cells/total_cells*100):.1f}%" if total_cells else "N/A")
    ]

    for col, (icon, label, value) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; text-align: center; color: white;">
                <div style="font-size: 24px;">{icon}</div>
                <div style="font-size: 12px; opacity: 0.9;">{label}</div>
                <div style="font-size: 20px; font-weight: bold;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()


def _render_overview_tab():
    state = st.session_state.app_state
    profiles = state.column_profiles
    df = state.df

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Type Distribution")
        dtype_counts = {}
        for p in profiles.values():
            t = 'Numeric' if any(x in p.dtype for x in ['int', 'float']) else 'DateTime' if 'date' in p.dtype else 'Text' if p.dtype == 'object' else 'Other'
            dtype_counts[t] = dtype_counts.get(t, 0) + 1

        fig = px.pie(values=list(dtype_counts.values()), names=list(dtype_counts.keys()),
                    hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Data Quality by Column")
        quality_data = []
        for col, p in profiles.items():
            comp = getattr(p, 'non_null_percentage', 100)
            uniq = p.unique_percentage
            quality_data.append({'Column': col, 'Completeness': comp, 'Uniqueness': uniq})

        if quality_data:
            qdf = pd.DataFrame(quality_data).sort_values('Completeness').head(15)
            fig = px.bar(qdf, x='Column', y=['Completeness', 'Uniqueness'],
                        barmode='group', color_discrete_sequence=['#10b981', '#3b82f6'])
            fig.update_layout(xaxis_tickangle=-45, yaxis_title="Percentage")
            st.plotly_chart(fig, use_container_width=True)

    # Data Volume & Missing Data
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Data by Column")
        null_data = [(c, p.null_percentage) for c, p in profiles.items() if p.null_percentage > 0]
        if null_data:
            null_df = pd.DataFrame(null_data, columns=['Column', 'Null %']).sort_values('Null %', ascending=True).tail(10)
            fig = px.bar(null_df, x='Null %', y='Column', orientation='h', color='Null %',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data!")

    with col2:
        st.subheader("Column Cardinality")
        card_data = [{'Column': c, 'Unique Values': p.unique_count, 'Type': 'Numeric' if any(x in p.dtype for x in ['int', 'float']) else 'Other'}
                     for c, p in profiles.items()]
        card_df = pd.DataFrame(card_data).sort_values('Unique Values', ascending=False).head(15)
        fig = px.scatter(card_df, x='Column', y='Unique Values', color='Type', size='Unique Values',
                        color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def _render_profiles_tab():
    state = st.session_state.app_state
    profiles = state.column_profiles
    df = state.df

    # Filters
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1: search = st.text_input("🔍 Search columns")
    with c2: dtype_filter = st.selectbox("Type", ["All", "Numeric", "Text", "Date", "High Risk"])
    with c3: sort_by = st.selectbox("Sort", ["Name", "Null %", "Uniqueness", "Risk"])

    filtered = {k: v for k, v in profiles.items() if not search or search.lower() in k.lower()}

    if dtype_filter == "Numeric":
        filtered = {k: v for k, v in filtered.items() if any(t in v.dtype for t in ['int', 'float'])}
    elif dtype_filter == "Text":
        filtered = {k: v for k, v in filtered.items() if v.dtype == 'object'}
    elif dtype_filter == "Date":
        filtered = {k: v for k, v in filtered.items() if 'date' in v.dtype}
    elif dtype_filter == "High Risk":
        filtered = {k: v for k, v in filtered.items() if getattr(v, 'risk_level', 'Low') == 'High'}

    if sort_by == "Null %":
        filtered = dict(sorted(filtered.items(), key=lambda x: x[1].null_percentage, reverse=True))
    elif sort_by == "Uniqueness":
        filtered = dict(sorted(filtered.items(), key=lambda x: x[1].unique_percentage, reverse=True))
    elif sort_by == "Risk":
        filtered = dict(sorted(filtered.items(), key=lambda x: getattr(x[1], 'risk_score', 0), reverse=True))

    st.write(f"Showing {len(filtered)} of {len(profiles)} columns")

    # Display as cards
    for col, prof in list(filtered.items())[:30]:
        with st.expander(f"📊 {col} | {prof.dtype} | Quality: {100-prof.null_percentage:.0f}%"):
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("**Volume**")
                st.write(f"Rows: {prof.total_rows:,}")
                st.write(f"Non-null: {getattr(prof, 'non_null_count', prof.total_rows - prof.null_count):,}")
                st.write(f"Null: {prof.null_count:,} ({prof.null_percentage:.1f}%)")

            with c2:
                st.markdown("**Uniqueness**")
                dup = prof.total_rows - prof.unique_count
                st.write(f"Unique: {prof.unique_count:,}")
                st.write(f"Duplicates: {dup:,}")
                st.write(f"Unique %: {prof.unique_percentage:.1f}%")

            with c3:
                st.markdown("**Length**")
                st.write(f"Min: {getattr(prof, 'min_length', 'N/A')}")
                st.write(f"Max: {getattr(prof, 'max_length', 'N/A')}")
                st.write(f"Avg: {getattr(prof, 'avg_length', 0):.1f}")

            with c4:
                st.markdown("**Risk**")
                risk = getattr(prof, 'risk_level', 'Low')
                color = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
                st.write(f"Level: {color} {risk}")
                st.write(f"Score: {getattr(prof, 'risk_score', 0)}/100")

            # Pattern detections
            validations = analyze_column_validations(df, col)
            if validations:
                st.markdown("**🔍 Detected Formats**")
                for v in validations[:3]:
                    st.write(f"• **{v['type']}** ({v['category']}): {v['coverage']}% match ({v['valid_count']} valid, {v['invalid_count']} invalid)")

            # Duplicate analysis
            if prof.unique_count < prof.total_rows:
                dups = find_duplicate_groups(df, col)
                if dups:
                    with st.expander(f"⚠️ Duplicates ({len(dups)} groups)"):
                        for d in dups[:5]:
                            st.write(f"- Value '{d['value'][:30]}' appears {d['count']} times ({d['percentage']}%)")


def _render_validations_tab():
    state = st.session_state.app_state
    df = state.df

    st.subheader("Universal Validation Detection")

    with st.spinner("Analyzing all validation patterns..."):
        validations = detect_all_validations(df)

    if not validations:
        st.info("No standard validation patterns detected")
        return

    # Summary metrics
    cats = set(v['Category'] for v in validations)
    countries = set(v['Country'] for v in validations)

    m1, m2, m3 = st.columns(3)
    m1.metric("Validations Found", len(validations))
    m2.metric("Categories", len(cats))
    m3.metric("Countries", len(countries))

    # Coverage chart
    st.subheader("Validation Coverage by Column")
    val_df = pd.DataFrame(validations)
    fig = px.scatter(val_df, x='Column', y='Validation Type', size='Coverage %', color='Category',
                    hover_data=['Valid Count', 'Invalid Count', 'Format'])
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Validation Details")
    display_df = val_df[['Column', 'Validation Type', 'Category', 'Country', 'Coverage %', 'Valid Count', 'Invalid Count', 'Format', 'Example']]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Invalid samples
    st.subheader("Invalid Format Samples")
    for v in validations:
        if v['Invalid Samples']:
            with st.expander(f"⚠️ {v['Column']} - {v['Validation Type']} ({len(v['Invalid Samples'])} samples)"):
                st.write("Expected format:", v['Format'])
                st.write("Invalid examples:", v['Invalid Samples'])


def _render_match_rules_tab():
    state = st.session_state.app_state

    st.subheader("🎯 Match Rule Suggestions")

    rules = generate_match_rules(state.df, state.column_profiles)

    if not rules:
        st.warning("No match rules could be generated")
        return

    # Rule distribution
    types = {}
    probs = {}
    for r in rules:
        types[r['Rule Type']] = types.get(r['Rule Type'], 0) + 1
        probs[r['Match Probability']] = probs.get(r['Match Probability'], 0) + 1

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(values=list(types.values()), names=list(types.keys()), title="Rule Types")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(x=list(probs.keys()), y=list(probs.values()), title="Match Probability Distribution",
                    color=list(probs.keys()), color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)

    # Rules table
    st.subheader("Suggested Match Rules")
    rules_df = pd.DataFrame(rules)

    # Color code by probability
    def color_prob(val):
        if val in ['Strongest', 'Very Strong', 'Enterprise']: return 'background-color: #10b981; color: white'
        elif val in ['Strong', 'Good', 'Fashion Strong']: return 'background-color: #3b82f6; color: white'
        else: return 'background-color: #f59e0b; color: white'

    styled_df = rules_df[['Rule No', 'Rule Type', 'Columns', 'Match Probability', 'Rationale']].style.applymap(color_prob, subset=['Match Probability'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Rule details
    st.subheader("Rule Details")
    for rule in rules:
        with st.expander(f"{rule['Rule No']}: {rule['Rule Type']} Match on {rule['Columns']}"):
            st.write(f"**Probability:** {rule['Match Probability']}")
            st.write(f"**Confidence Score:** {rule.get('Confidence', 'N/A')}")
            st.write(f"**Rationale:** {rule['Rationale']}")


def _render_issues_tab():
    state = st.session_state.app_state

    st.subheader("⚠️ Data Quality Issues")

    dq_rules = generate_dq_rules(state.df, state.column_profiles)

    if not dq_rules:
        st.success("✅ No quality issues detected!")
        return

    # Severity distribution
    sev_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for r in dq_rules:
        sev_counts[r['Severity']] = sev_counts.get(r['Severity'], 0) + 1

    col1, col2 = st.columns([1, 2])

    with col1:
        fig = px.pie(values=list(sev_counts.values()), names=list(sev_counts.keys()),
                    title="Issues by Severity",
                    color=list(sev_counts.keys()),
                    color_discrete_map={'CRITICAL': '#dc2626', 'HIGH': '#ea580c', 'MEDIUM': '#ca8a04', 'LOW': '#16a34a'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Issues by dimension
        dim_counts = {}
        for r in dq_rules:
            dim_counts[r['Dimension']] = dim_counts.get(r['Dimension'], 0) + 1
        fig = px.bar(x=list(dim_counts.keys()), y=list(dim_counts.values()),
                    title="Issues by DQ Dimension", color=list(dim_counts.keys()))
        st.plotly_chart(fig, use_container_width=True)

    # Issues table
    st.subheader("Quality Rules & Actions")
    issues_df = pd.DataFrame(dq_rules)

    def color_severity(val):
        colors = {'CRITICAL': '#dc2626', 'HIGH': '#ea580c', 'MEDIUM': '#ca8a04', 'LOW': '#16a34a'}
        return f'background-color: {colors.get(val, "gray")}; color: white'

    styled = issues_df[['Column', 'Dimension', 'Type', 'Severity', 'Found', 'Expected', 'Action']].style.applymap(color_severity, subset=['Severity'])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_export_tab():
    state = st.session_state.app_state

    st.subheader("📥 Export Profiling Report")

    fmt = st.radio("Format", ["Excel (.xlsx)", "JSON (.json)"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
            _generate_excel_report()
    with col2:
        if st.button("📋 Generate JSON Report", type="primary", use_container_width=True):
            _generate_json_report()

# ==========================================
# EXPORT FUNCTIONS (UNCHANGED)
# ==========================================

def _analyze_special_chars_detailed(df):
    data = []
    for col in df.select_dtypes(include=['object']).columns:
        counter = Counter()
        for val in df[col].dropna().astype(str):
            for char in set(val):
                if ord(char) > 127 or (not char.isalnum() and not char.isspace()):
                    counter[char] += val.count(char)
        for char, count in counter.most_common():
            try: uname = unicodedata.name(char)
            except: uname = "UNKNOWN"
            data.append({'Column': col, 'Character': char, 'Unicode Name': uname, 'Count': count})
    return data


def _generate_excel_report():
    state = st.session_state.app_state
    progress = st.progress(0)
    output = io.BytesIO()

    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df = state.df
            profiles = state.column_profiles

            # 1. Executive Summary
            progress.progress(10)
            total_missing = sum(p.null_count for p in profiles.values())
            quality = sum(getattr(p, 'non_null_percentage', 100) for p in profiles.values()) / len(profiles) if profiles else 0

            pd.DataFrame({
                'Metric': ['Total Rows', 'Total Columns', 'Total Cells', 'Missing Cells', 'Completeness %', 'Quality Score', 'Generated At'],
                'Value': [len(df), len(df.columns), len(df)*len(df.columns), total_missing, 
                         f"{((len(df)*len(df.columns)-total_missing)/(len(df)*len(df.columns))*100):.2f}%",
                         f"{quality:.1f}%", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            }).to_excel(writer, sheet_name='Executive Summary', index=False)

            # 2. Column Profiles (Enhanced)
            progress.progress(25)
            profile_data = []
            for col, p in profiles.items():
                dup_count = p.total_rows - p.unique_count
                profile_data.append({
                    'Column Name': col,
                    'Data Type': p.dtype,
                    'Total Rows': p.total_rows,
                    'Non-Null Count': p.total_rows - p.null_count,
                    'Null Count': p.null_count,
                    'Null Percentage': f"{p.null_percentage:.2f}%",
                    'Unique Count': p.unique_count,
                    'Duplicate Count': dup_count,
                    'Unique Percentage': f"{p.unique_percentage:.2f}%",
                    'Min Length': getattr(p, 'min_length', 'N/A'),
                    'Max Length': getattr(p, 'max_length', 'N/A'),
                    'Avg Length': f"{getattr(p, 'avg_length', 0):.2f}",
                    'Risk Level': getattr(p, 'risk_level', 'Low'),
                    'Risk Score': getattr(p, 'risk_score', 0)
                })
            pd.DataFrame(profile_data).to_excel(writer, sheet_name='Column Profiles', index=False)

            # 3. Special Characters
            progress.progress(40)
            chars = _analyze_special_chars_detailed(df)
            (pd.DataFrame(chars) if chars else pd.DataFrame({'Message': ['No special characters found']})).to_excel(
                writer, sheet_name='Special Characters', index=False)

            # 4. Pattern Found (Enhanced with validations)
            progress.progress(55)
            validations = detect_all_validations(df)
            if validations:
                val_df = pd.DataFrame(validations)
                val_df['Valid Samples'] = val_df['Valid Samples'].apply(lambda x: ', '.join(x) if x else '')
                val_df['Invalid Samples'] = val_df['Invalid Samples'].apply(lambda x: ', '.join(x) if x else '')
                val_df.to_excel(writer, sheet_name='Pattern Found', index=False)
            else:
                pd.DataFrame({'Message': ['No patterns detected']}).to_excel(writer, sheet_name='Pattern Found', index=False)

            # 5. Match Rule Suggestion
            progress.progress(75)
            match_rules = generate_match_rules(df, profiles)
            pd.DataFrame(match_rules).to_excel(writer, sheet_name='Match Rule Suggestion', index=False)

            # 6. Data Quality and Validation Rules
            progress.progress(90)
            dq_rules = generate_dq_rules(df, profiles)
            pd.DataFrame(dq_rules).to_excel(writer, sheet_name='Data Quality and Validation Rules', index=False)

        progress.progress(100)
        st.download_button("⬇️ Download Excel Report", data=output.getvalue(),
                          file_name=f"data_profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        show_toast("Excel report generated!", "success")
    except Exception as e:
        st.error(f"Error generating Excel: {str(e)}")


def _generate_json_report():
    import json
    state = st.session_state.app_state

    try:
        df = state.df
        profiles = state.column_profiles

        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_rows': len(df),
                'total_columns': len(df.columns)
            },
            'executive_summary': {
                'missing_cells': sum(p.null_count for p in profiles.values()),
                'completeness': f"{sum(getattr(p, 'non_null_percentage', 100) for p in profiles.values())/len(profiles):.2f}%" if profiles else "N/A"
            },
            'column_profiles': {col: {
                'type': p.dtype,
                'null_percentage': p.null_percentage,
                'unique_percentage': p.unique_percentage,
                'duplicate_count': p.total_rows - p.unique_count
            } for col, p in profiles.items()},
            'validations': detect_all_validations(df),
            'match_rules': generate_match_rules(df, profiles),
            'dq_rules': generate_dq_rules(df, profiles)
        }

        st.download_button("⬇️ Download JSON Report", data=json.dumps(report, indent=2, default=str),
                          file_name=f"data_profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                          mime="application/json")
        show_toast("JSON report generated!", "success")
    except Exception as e:
        st.error(f"Error generating JSON: {str(e)}")