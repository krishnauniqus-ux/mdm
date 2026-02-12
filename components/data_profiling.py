"""Data Profiling Component - AI-Powered Dynamic Validation with Azure OpenAI"""

import streamlit as st
import pandas as pd
import io
import re
import json
import unicodedata
from datetime import datetime
from collections import Counter, defaultdict
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import hashlib

# Azure OpenAI imports
try:
    from openai import AzureOpenAI
except ImportError:
    st.error("Please install: pip install openai")

from state.session import st, show_toast
from core.profiler import DataProfilerEngine

# ==========================================
# AZURE OPENAI CONFIGURATION
# ==========================================

class AzureOpenAIConfig:
    """Azure OpenAI Configuration - Set these in environment variables or secrets"""
    AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = st.secrets.get("AZURE_OPENAI_KEY")
    AZURE_OPENAI_DEPLOYMENT = st.secrets.get("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION")
     
    @classmethod
    def validate(cls):
        missing = []
        if not cls.AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not cls.AZURE_OPENAI_KEY:
            missing.append("AZURE_OPENAI_KEY")
        if not cls.AZURE_OPENAI_DEPLOYMENT:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        return missing

# ==========================================
# AI-POWERED VALIDATION ENGINE
# ==========================================

class AIValidationEngine:
    """Enterprise-grade AI-powered validation rule generator using Azure OpenAI"""
    
    # Strict DQ Dimensions as per enterprise standards
    DQ_DIMENSIONS = [
        "Accuracy", "Completeness", "Consistency", "Validity", 
        "Uniqueness", "Timeliness", "Integrity", "Conformity",
        "Reliability", "Relevance", "Precision", "Accessibility"
    ]
    
    def __init__(self):
        self.client = None
        self.cache = {}  # Cache for validation rules to avoid repeated API calls
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        try:
            missing = AzureOpenAIConfig.validate()
            if missing:
                st.warning(f"Azure OpenAI not configured. Missing: {', '.join(missing)}")
                return
            
            self.client = AzureOpenAI(
                azure_endpoint=AzureOpenAIConfig.AZURE_OPENAI_ENDPOINT,
                api_key=AzureOpenAIConfig.AZURE_OPENAI_KEY,
                api_version=AzureOpenAIConfig.AZURE_OPENAI_API_VERSION
            )
        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI: {str(e)}")
    
    def _get_cache_key(self, column_name: str, sample_data: List[str], data_type: str) -> str:
        """Generate cache key for column analysis"""
        data_hash = hashlib.md5(str(sample_data[:50]).encode()).hexdigest()
        return f"{column_name}_{data_type}_{data_hash}"
    
    def _call_azure_openai(self, messages: List[Dict], temperature: float = 0.1) -> Optional[str]:
        """Make Azure OpenAI API call with error handling"""
        if not self.client:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=AzureOpenAIConfig.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Azure OpenAI API Error: {str(e)}")
            return None
    
    def analyze_column_semantic_type(self, column_name: str, sample_data: List[str], 
                                    data_type: str, null_pct: float, unique_pct: float) -> Dict[str, Any]:
        """Use AI to determine the semantic type and validation rules for a column"""
        
        cache_key = self._get_cache_key(column_name, sample_data, data_type)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Prepare sample data for AI analysis
        samples = sample_data[:100] if sample_data else []
        sample_str = json.dumps(samples, indent=2)
        
        # HUMAN READABLE RULE PROMPT
        prompt = f"""Analyze this data column and generate human-readable validation rules.

COLUMN INFORMATION
==================
Column Name: {column_name}
Data Type: {data_type}
Sample Values: {sample_str}
Null Percentage: {null_pct:.1f}%
Unique Percentage: {unique_pct:.1f}%

REQUIRED OUTPUT FORMAT
======================
Return a JSON object with this exact structure:

{{
  "business_field_name": "Human-friendly field name (e.g., 'Posting Date', 'Email Address', 'Mobile Number')",
  "rules": [
    {{
      "dimension": "One of: Accuracy, Completeness, Consistency, Validity, Uniqueness, Timeliness, Integrity, Conformity, Reliability, Relevance, Precision, Accessibility",
      "rule_statement": "Human readable rule in format: [Field Name] + Must/Should + Business Condition. Example: 'Posting Date must not be future dated'"
    }}
  ]
}}

RULE WRITING GUIDELINES
=======================
Write rules in this exact format: [Field Name] + Must/Should + Business Condition

EXAMPLES BY FIELD TYPE:

Date Fields (Date of Birth, Posting Date, Invoice Date):
- Date of Birth must be a valid calendar date
- Date of Birth cannot be in the future
- Date of Birth cannot be more than 120 years in the past
- Posting Date must not be future dated
- Posting Date must be within the active financial period
- Invoice Date should not be blank

Email Fields (Email Address):
- Email Address must follow standard email format
- Email Address must contain @ symbol
- Email Address must contain valid domain name
- Email Address should not contain spaces
- Email Address must be unique for each customer

Phone Fields (Mobile Number, Phone):
- Mobile Number must contain only digits
- Mobile Number must have 10 digits
- Mobile Number must start with valid prefix (6,7,8,9)
- Mobile Number should not contain special characters

ID Fields (PAN, GST, Asset ID):
- PAN must follow government issued format
- PAN must be 10 characters
- PAN must be alphanumeric
- Asset ID must follow organization coding standard

Amount Fields (Amount, Cost, Price):
- Amount must be numeric
- Amount should not be negative
- Amount must have maximum 2 decimal places
- Amount should not exceed defined business limit

Text Fields (Description, Name):
- Description should not be blank
- Description should not contain only special characters
- Description should not contain generic values like "Test" or "NA"

REQUIREMENTS
============
1. Generate 1-3 meaningful rules per column
2. Use exact format: [Field Name] + Must/Should + Business Condition
3. Rules must be immediately understandable by business users
4. No technical jargon, regex, or code references
5. Each rule should be actionable and specific
6. Assign most appropriate DQ dimension

Return ONLY valid JSON, no markdown."""

        messages = [
            {"role": "system", "content": "You are a Business Data Analyst who writes clear, human-readable data quality rules for enterprise systems. Return only JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_azure_openai(messages, temperature=0.1)
        
        if response:
            try:
                # Clean response - remove markdown code blocks if present
                cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
                result = json.loads(cleaned)
                self.cache[cache_key] = result
                return result
            except json.JSONDecodeError as e:
                st.warning(f"Failed to parse AI response for {column_name}: {str(e)}")
        
        # Fallback to basic analysis if AI fails
        return self._fallback_analysis(column_name, sample_data, data_type, null_pct, unique_pct)
    
    def _fallback_analysis(self, column_name: str, sample_data: List[str], 
                          data_type: str, null_pct: float, unique_pct: float) -> Dict[str, Any]:
        """Basic fallback analysis when AI is unavailable"""
        rules = []
        field_name = column_name.replace('_', ' ').title()
        
        # Completeness rule for nulls
        if null_pct > 0:
            rules.append({
                "dimension": "Completeness",
                "rule_statement": f"{field_name} should not be blank or null"
            })
        
        # Uniqueness rule for near-unique
        if unique_pct >= 95:
            rules.append({
                "dimension": "Uniqueness",
                "rule_statement": f"{field_name} must be unique"
            })
        
        # Validity rule based on data type
        if 'date' in data_type.lower():
            rules.append({
                "dimension": "Validity",
                "rule_statement": f"{field_name} must be a valid calendar date"
            })
            rules.append({
                "dimension": "Timeliness",
                "rule_statement": f"{field_name} must not be future dated"
            })
        elif 'int' in data_type.lower() or 'float' in data_type.lower():
            rules.append({
                "dimension": "Validity",
                "rule_statement": f"{field_name} must be numeric"
            })
        elif 'object' in data_type.lower():
            rules.append({
                "dimension": "Validity",
                "rule_statement": f"{field_name} must contain valid text"
            })
        
        return {
            "business_field_name": field_name,
            "rules": rules
        }
    
    def generate_dynamic_rules(self, df: pd.DataFrame, column_name: str, 
                              profile: Any) -> List[Dict[str, Any]]:
        """Generate comprehensive validation rules for a column using AI"""
        
        sample_data = df[column_name].dropna().astype(str).head(100).tolist()
        data_type = str(profile.dtype)
        null_pct = profile.null_percentage
        unique_pct = profile.unique_percentage
        
        # Get AI analysis
        ai_analysis = self.analyze_column_semantic_type(column_name, sample_data, data_type, null_pct, unique_pct)
        
        rules = []
        field_name = ai_analysis.get('business_field_name', column_name)
        
        # Process AI-generated validation rules
        for idx, rule in enumerate(ai_analysis.get('rules', [])):
            # Validate dimension is in allowed list
            dimension = rule.get('dimension', 'Validity')
            if dimension not in self.DQ_DIMENSIONS:
                dimension = 'Validity'
            
            validation_rule = {
                'S.No': len(rules) + 1,
                'Column': column_name,
                'Business Field': field_name,
                'Dimension': dimension,
                'Data Quality Rule': rule.get('rule_statement', 'No rule specified')
            }
            rules.append(validation_rule)
        
        return rules
    
    def validate_data_against_rules(self, df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
        """Validate dataframe against AI-generated rules and return results with examples"""
        validation_results = []
        
        for rule in rules:
            column = rule.get('Column')
            if column not in df.columns:
                continue
            
            rule_statement = rule.get('Data Quality Rule', '').lower()
            invalid_count = 0
            invalid_examples = []  # Store examples of invalid values
            
            # Simple validation logic based on rule statement keywords
            try:
                if 'not be blank' in rule_statement or 'not be null' in rule_statement:
                    mask = df[column].isna()
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).tolist()
                
                elif 'unique' in rule_statement:
                    mask = df[column].duplicated(keep=False)
                    invalid_count = mask.sum()
                    # Get duplicate values and their counts
                    dup_values = df[mask][column].value_counts().head(3)
                    invalid_examples = [f"{val} ({count} times)" for val, count in dup_values.items()]
                
                elif 'valid calendar date' in rule_statement or 'valid date' in rule_statement:
                    converted = pd.to_datetime(df[column], errors='coerce')
                    mask = converted.isna() & df[column].notna()
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).astype(str).tolist()
                
                elif 'not be future dated' in rule_statement or 'not be in the future' in rule_statement:
                    try:
                        dates = pd.to_datetime(df[column], errors='coerce')
                        mask = dates > datetime.now()
                        invalid_count = mask.sum()
                        invalid_examples = df[mask][column].head(5).astype(str).tolist()
                    except:
                        invalid_count = 0
                
                elif 'numeric' in rule_statement:
                    converted = pd.to_numeric(df[column], errors='coerce')
                    mask = converted.isna() & df[column].notna()
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).astype(str).tolist()
                
                elif 'not be negative' in rule_statement:
                    try:
                        numeric_vals = pd.to_numeric(df[column], errors='coerce')
                        mask = numeric_vals < 0
                        invalid_count = mask.sum()
                        invalid_examples = df[mask][column].head(5).astype(str).tolist()
                    except:
                        invalid_count = 0
                
                elif 'only digits' in rule_statement:
                    mask = ~df[column].astype(str).str.match(r'^\d+$')
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).astype(str).tolist()
                
                elif '10 digits' in rule_statement or '10 characters' in rule_statement:
                    mask = ~df[column].astype(str).str.len().eq(10)
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).astype(str).tolist()
                
                elif 'email' in rule_statement:
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    mask = ~df[column].astype(str).str.match(email_pattern)
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).astype(str).tolist()
                
                elif 'alphanumeric' in rule_statement:
                    mask = ~df[column].astype(str).str.match(r'^[a-zA-Z0-9]+$')
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).astype(str).tolist()
                
                elif 'not contain' in rule_statement and 'special' in rule_statement:
                    mask = df[column].astype(str).str.contains(r'[^a-zA-Z0-9\s]', regex=True, na=False)
                    invalid_count = mask.sum()
                    invalid_examples = df[mask][column].head(5).astype(str).tolist()
                
                else:
                    # Generic pattern matching for other rules
                    invalid_count = 0
                    invalid_examples = []
                    
            except Exception as e:
                invalid_count = 0
                invalid_examples = []
            
            # Format examples for display
            if invalid_count == 0:
                examples_str = "✓ All values valid - No issues found"
            elif invalid_examples:
                # Clean and truncate examples
                cleaned_examples = []
                for ex in invalid_examples:
                    ex_str = str(ex).strip()
                    if len(ex_str) > 50:
                        ex_str = ex_str[:47] + "..."
                    cleaned_examples.append(ex_str)
                examples_str = "; ".join(cleaned_examples)
            else:
                examples_str = f"{invalid_count} issues found (examples not captured)"
            
            validation_results.append({
                'Column': column,
                'Dimension': rule.get('Dimension'),
                'Invalid_Count': int(invalid_count),
                'Issues_Found_Example': examples_str
            })
        
        return pd.DataFrame(validation_results)


# ==========================================
# ENHANCED VALIDATION DETECTION
# ==========================================

class DynamicValidationDetector:
    """Enhanced validation detection combining AI and statistical analysis"""
    
    def __init__(self):
        self.ai_engine = AIValidationEngine()
    
    def detect_all_validations(self, df: pd.DataFrame, profiles: Dict) -> List[Dict]:
        """Detect all validations using AI-powered analysis"""
        all_validations = []
        
        progress_bar = st.progress(0)
        total_cols = len(df.columns)
        
        for idx, col in enumerate(df.columns):
            progress_bar.progress((idx + 1) / total_cols)
            
            if col not in profiles:
                continue
            
            profile = profiles[col]
            
            # Skip columns with too many nulls
            if profile.null_percentage > 90:
                continue
            
            # Generate AI-powered rules
            try:
                rules = self.ai_engine.generate_dynamic_rules(df, col, profile)
                all_validations.extend(rules)
            except Exception as e:
                st.warning(f"Error analyzing {col}: {str(e)}")
        
        progress_bar.empty()
        return all_validations
    
    def generate_comprehensive_dq_rules(self, df: pd.DataFrame, profiles: Dict) -> pd.DataFrame:
        """Generate comprehensive DQ rules with validation results"""
        all_rules = self.detect_all_validations(df, profiles)
        
        # Validate data against rules
        validation_results = self.ai_engine.validate_data_against_rules(df, all_rules)
        
        # Create comprehensive output - HUMAN READABLE FORMAT
        output_data = []
        for idx, rule in enumerate(all_rules, 1):
            # Find matching validation result
            match = validation_results[
                (validation_results['Column'] == rule.get('Column')) & 
                (validation_results['Dimension'] == rule.get('Dimension'))
            ]
            
            invalid_count = match.iloc[0]['Invalid_Count'] if not match.empty else 0
            issues_example = match.iloc[0]['Issues_Found_Example'] if not match.empty else "✓ All values valid - No issues found"
            
            row = {
                'S.No': idx,
                'Column': rule.get('Column'),
                'Business Field': rule.get('Business Field'),
                'Dimension': rule.get('Dimension'),
                'Data Quality Rule': rule.get('Data Quality Rule'),
                'Issues Found': invalid_count,
                'Issues Found Example': issues_example
            }
            
            output_data.append(row)
        
        return pd.DataFrame(output_data)


# ==========================================
# CORE FUNCTIONS (REFACTORED)
# ==========================================

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


def safe_get_special_chars(prof):
    try:
        if hasattr(prof, 'special_chars') and prof.special_chars:
            return [c for c in prof.special_chars if isinstance(c, dict) and 'count' in c]
    except: 
        pass
    return []


def _remove_ghost_columns():
    state = st.session_state.app_state
    if state.df is None: 
        return False
    ghosts = [c for c in state.df.columns if str(c).startswith('Unnamed:') and state.df[c].isnull().all()]
    if ghosts:
        state.df = state.df.drop(columns=ghosts)
        for c in ghosts: 
            state.column_profiles.pop(c, None)
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


    # Executive Dashboard
    _render_executive_dashboard()

    # Main Tabs
    tabs = st.tabs([
        "📊 Overview", 
        "📋 Column Profiles", 
        "🤖 AI Validations", 
        "🎯 Match Rules", 
        "📥 Export"
    ])

    with tabs[0]: 
        _render_overview_tab()
    with tabs[1]: 
        _render_profiles_tab()
    with tabs[2]: 
        _render_ai_validations_tab()
    with tabs[3]: 
        _render_match_rules_tab()
    with tabs[4]: 
        _render_export_tab()


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
        
        if card_data:
            card_df = pd.DataFrame(card_data).sort_values('Unique Values', ascending=False).head(15)
            fig = px.scatter(card_df, x='Column', y='Unique Values', color='Type', size='Unique Values',
                            color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cardinality data available")


def _render_profiles_tab():
    state = st.session_state.app_state
    profiles = state.column_profiles
    df = state.df

    # Filters
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1: 
        search = st.text_input("🔍 Search columns")
    with c2: 
        dtype_filter = st.selectbox("Type", ["All", "Numeric", "Text", "Date", "High Risk"])
    with c3: 
        sort_by = st.selectbox("Sort", ["Name", "Null %", "Uniqueness", "Risk"])

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
    # show collabsible in grid view
    
    cols_per_row = 2
    items = list(filtered.items())[:30]

    for i in range(0, len(items), cols_per_row):

        cols = st.columns(cols_per_row)

        for col_ui, (col, prof) in zip(cols, items[i:i+cols_per_row]):

            with col_ui:

                with st.expander(
                    f"📊 {col} | {prof.dtype} | Quality: {100-prof.null_percentage:.0f}%"
                ):

                    c1, c2, c3, c4 = st.columns(4)

                    # Volume
                    with c1:
                        st.markdown("**Volume**")
                        st.write(f"Rows: {prof.total_rows:,}")
                        st.write(f"Non-null: {getattr(prof, 'non_null_count', prof.total_rows - prof.null_count):,}")
                        st.write(f"Null: {prof.null_count:,} ({prof.null_percentage:.1f}%)")

                    # Uniqueness
                    with c2:
                        st.markdown("**Uniqueness**")
                        dup = prof.total_rows - prof.unique_count
                        st.write(f"Unique: {prof.unique_count:,}")
                        st.write(f"Duplicates: {dup:,}")
                        st.write(f"Unique %: {prof.unique_percentage:.1f}%")

                    # Length
                    with c3:
                        st.markdown("**Length**")
                        st.write(f"Min: {getattr(prof, 'min_length', 'N/A')}")
                        st.write(f"Max: {getattr(prof, 'max_length', 'N/A')}")
                        st.write(f"Avg: {getattr(prof, 'avg_length', 0):.1f}")

                    # Risk
                    with c4:
                        st.markdown("**Risk**")
                        risk = getattr(prof, 'risk_level', 'Low')
                        color = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
                        st.write(f"Level: {color} {risk}")
                        st.write(f"Score: {getattr(prof, 'risk_score', 0)}/100")

                    # Duplicate analysis (nested collapsible)
                    if prof.unique_count < prof.total_rows:
                        dups = find_duplicate_groups(df, col)
                        if dups:
                            with st.expander(f"⚠️ Duplicates ({len(dups)} groups)"):
                                for d in dups[:5]:
                                    st.write(
                                        f"- Value '{d['value'][:30]}' "
                                        f"appears {d['count']} times ({d['percentage']}%)"
                                    )


def _render_ai_validations_tab():
    """NEW: AI-Powered Validations Tab - HUMAN READABLE FORMAT"""
    state = st.session_state.app_state
    df = state.df
    profiles = state.column_profiles
    
    # Check Azure OpenAI configuration
    missing_config = AzureOpenAIConfig.validate()
    if missing_config:
        st.error(f"⚠️ Azure OpenAI not configured. Missing: {', '.join(missing_config)}")
        st.info("Please set these in your Streamlit secrets or environment variables.")
        return
    
    # Rules are now stored in state.ai_validation_rules and state.ai_validation_rules_generated
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if not state.ai_validation_rules_generated:
            if st.button("🚀 Generate AI Validation Rules", type="primary"):
                with st.spinner("🤖 Analyzing data with Azure OpenAI..."):
                    detector = DynamicValidationDetector()
                    validation_df = detector.generate_comprehensive_dq_rules(df, profiles)
                    
                    # Store in state for persistence
                    state.ai_validation_rules = validation_df
                    state.ai_validation_rules_generated = True
                    
                    # Explicitly save to disk
                    from state.session import _save_persisted_data
                    _save_persisted_data()
                    
                    st.success(f"✅ Generated {len(validation_df)} validation rules!")
                    st.rerun()
        else:
            if st.button("🗑️ Clear Generated Rules", type="secondary"):
                state.ai_validation_rules_generated = False
                state.ai_validation_rules = pd.DataFrame()
                
                # Explicitly save to disk
                from state.session import _save_persisted_data
                _save_persisted_data()
                st.rerun()
    
    
    # Use state data for display
    if state.ai_validation_rules_generated and state.ai_validation_rules is not None and not state.ai_validation_rules.empty:
        validation_df = state.ai_validation_rules
        # Show record count
        st.caption(f"Showing {len(validation_df)} validation rules")
        
        # Filter by Dimension
        available_dims = validation_df['Dimension'].unique().tolist()
        selected_dims = st.multiselect(
            "Filter by DQ Dimension", 
            available_dims, 
            default=available_dims
        )
        
        if selected_dims:
            filtered_df = validation_df[validation_df['Dimension'].isin(selected_dims)]
        else:
            filtered_df = validation_df
        
        # Display columns - CLEAN HUMAN READABLE WITH EXAMPLES
        display_cols = ['S.No', 'Business Field', 'Dimension', 'Data Quality Rule', 'Issues Found', 'Issues Found Example']
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Style the dataframe
        def highlight_issues(val):
            if isinstance(val, int) and val > 0:
                return 'background-color: #fee2e2; color: #991b1b; font-weight: bold;'
            return ''
        
        def highlight_examples(val):
            if isinstance(val, str) and val.startswith('✓'):
                return 'background-color: #dcfce7; color: #166534; font-size: 11px; font-style: italic;'
            elif isinstance(val, str) and len(val) > 0:
                return 'background-color: #fef3c7; color: #92400e; font-size: 11px;'
            return ''
        
        def highlight_dimension(val):
            colors = {
                'Accuracy': 'background-color: #dbeafe; color: #1e40af',
                'Completeness': 'background-color: #dcfce7; color: #166534',
                'Consistency': 'background-color: #fef3c7; color: #92400e',
                'Validity': 'background-color: #fce7f3; color: #9d174d',
                'Uniqueness': 'background-color: #f3e8ff; color: #6b21a8',
                'Timeliness': 'background-color: #ccfbf1; color: #0f766e',
                'Integrity': 'background-color: #fee2e2; color: #991b1b',
                'Conformity': 'background-color: #e0e7ff; color: #3730a3',
                'Reliability': 'background-color: #ffedd5; color: #9a3412',
                'Relevance': 'background-color: #ecfccb; color: #3f6212',
                'Precision': 'background-color: #fae8ff; color: #86198f',
                'Accessibility': 'background-color: #e0f2fe; color: #075985'
            }
            return colors.get(val, '')
        
        # Apply styling and display - use actual row count for height
        row_count = len(filtered_df)
        # Calculate height: min 100px, max 600px, approx 35px per row
        table_height = min(max(row_count * 35 + 50, 100), 600)
        
        styled_df = filtered_df[display_cols].style\
            .applymap(highlight_dimension, subset=['Dimension'])\
            .applymap(highlight_issues, subset=['Issues Found'])\
            .applymap(highlight_examples, subset=['Issues Found Example'])
        
        st.dataframe(styled_df, use_container_width=True, height=table_height)
        
        # Summary statistics
        dim_counts = filtered_df['Dimension'].value_counts().reset_index()
        dim_counts.columns = ['Dimension', 'Rule Count']
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(dim_counts, x='Dimension', y='Rule Count', 
                        color='Dimension', title="Rules by DQ Dimension")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(dim_counts, values='Rule Count', names='Dimension',
                        title="DQ Dimension Distribution")
            st.plotly_chart(fig, use_container_width=True)
        


def _render_match_rules_tab():
    state = st.session_state.app_state
    rules = generate_match_rules(state.df, state.column_profiles)

    if not rules:
        st.warning("No match rules could be generated")
        return

    # Rule distribution
    types = {}
    probs = {}

    # Rules table
    st.subheader("Suggested Match Rules")
    rules_df = pd.DataFrame(rules)

    # Color code by probability
    def color_prob(val):
        if val in ['Strongest', 'Very Strong', 'Enterprise']: 
            return 'background-color: #10b981; color: white'
        elif val in ['Strong', 'Good', 'Fashion Strong']: 
            return 'background-color: #3b82f6; color: white'
        else: 
            return 'background-color: #f59e0b; color: white'

    styled_df = rules_df[['Rule No', 'Rule Type', 'Columns', 'Match Probability', 'Rationale']].style.applymap(
        color_prob, subset=['Match Probability']
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Rule details
    st.subheader("Rule Details")
    for rule in rules:
        with st.expander(f"{rule['Rule No']}: {rule['Rule Type']} Match on {rule['Columns']}"):
            st.write(f"**Probability:** {rule['Match Probability']}")
            st.write(f"**Confidence Score:** {rule.get('Confidence', 'N/A')}")
            st.write(f"**Rationale:** {rule['Rationale']}")




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
# EXPORT FUNCTIONS (ENHANCED)
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
            try: 
                uname = unicodedata.name(char)
            except: 
                uname = "UNKNOWN"
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

            # 4. AI Validation Rules - HUMAN READABLE FORMAT WITH EXAMPLES
            progress.progress(55)
            if state.ai_validation_rules_generated and state.ai_validation_rules is not None and not state.ai_validation_rules.empty:
                validation_df = state.ai_validation_rules.copy()
                # Clean column order - HUMAN READABLE WITH EXAMPLES
                col_order = ['S.No', 'Business Field', 'Dimension', 'Data Quality Rule', 'Issues Found', 'Issues Found Example']
                col_order = [c for c in col_order if c in validation_df.columns]
                validation_df = validation_df[col_order]
                validation_df.to_excel(writer, sheet_name='Data Quality Rules', index=False)
            else:
                pd.DataFrame({'Message': ['No AI validation rules generated. Run validation analysis first.']}).to_excel(
                    writer, sheet_name='Data Quality Rules', index=False)

            # 5. Match Rule Suggestion
            progress.progress(75)
            match_rules = generate_match_rules(df, profiles)
            pd.DataFrame(match_rules).to_excel(writer, sheet_name='Match Rules', index=False)

            # 6. DQ Dimension Summary
            progress.progress(90)
            if state.ai_validation_rules_generated and state.ai_validation_rules is not None and not state.ai_validation_rules.empty:
                dq_summary = state.ai_validation_rules.groupby('Dimension').agg({
                    'S.No': 'count',
                    'Issues Found': 'sum'
                }).reset_index()
                dq_summary.columns = ['Dimension', 'Rule Count', 'Total Issues']
                dq_summary = dq_summary.sort_values('Rule Count', ascending=False)
                dq_summary.to_excel(writer, sheet_name='DQ Summary', index=False)
            else:
                pd.DataFrame({'Message': ['No data quality analysis available']}).to_excel(
                    writer, sheet_name='DQ Summary', index=False)

        progress.progress(100)
        st.download_button(
            "⬇️ Download Excel Report", 
            data=output.getvalue(),
            file_name=f"Data_Profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
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
                'total_columns': len(df.columns),
                'ai_powered': True,
                'dq_dimensions': AIValidationEngine.DQ_DIMENSIONS
            },
            'executive_summary': {
                'missing_cells': sum(p.null_count for p in profiles.values()),
                'completeness': f"{sum(getattr(p, 'non_null_percentage', 100) for p in profiles.values())/len(profiles):.2f}%" if profiles else "N/A"
            },
            'column_profiles': {
                col: {
                    'type': p.dtype,
                    'null_percentage': p.null_percentage,
                    'unique_percentage': p.unique_percentage,
                    'duplicate_count': p.total_rows - p.unique_count
                } for col, p in profiles.items()
            },
            'data_quality_rules': state.ai_validation_rules.to_dict('records') if state.ai_validation_rules_generated and state.ai_validation_rules is not None else [],
            'match_rules': generate_match_rules(df, profiles)
        }

        st.download_button(
            "⬇️ Download JSON Report", 
            data=json.dumps(report, indent=2, default=str),
            file_name=f"dq_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        show_toast("JSON report generated!", "success")
    except Exception as e:
        st.error(f"Error generating JSON: {str(e)}")