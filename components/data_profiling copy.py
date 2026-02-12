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
                                    data_type: str) -> Dict[str, Any]:
        """Use AI to determine the semantic type and validation rules for a column"""
        
        cache_key = self._get_cache_key(column_name, sample_data, data_type)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Prepare sample data for AI analysis
        samples = sample_data[:20] if sample_data else []
        sample_str = json.dumps(samples, indent=2)
        
        prompt = f"""Analyze this data column and provide intelligent validation rules.

Column Name: {column_name}
Data Type: {data_type}
Sample Values (first 20): {sample_str}

Analyze and return a JSON object with:
1. semantic_type: What type of data this represents (e.g., "Email Address", "Phone Number", "Product Code", "Customer ID", "Date", "Currency", "Percentage", etc.)
2. category: Broader category (Contact, Financial, Government ID, Technical, Date, Location, Product, Personal Info, etc.)
3. country_context: Likely country/region context (India, USA, UK, Global, etc.)
4. validation_rules: Array of validation objects with:
   - rule_name: Name of the validation
   - condition: Python regex pattern or logical condition
   - condition_type: "regex", "range", "length", "enum", "custom"
   - description: Human-readable description
   - severity: "CRITICAL", "HIGH", "MEDIUM", "LOW"
   - example_valid: Example of valid value
   - example_invalid: Example of invalid value
5. data_quality_checks: Array of DQ dimension checks (Completeness, Uniqueness, Validity, Consistency, Accuracy, Timeliness)
6. confidence_score: 0-100 confidence in this classification

Return ONLY valid JSON, no markdown formatting."""

        messages = [
            {"role": "system", "content": "You are an expert data governance and validation specialist. Analyze data patterns and generate enterprise-grade validation rules. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_azure_openai(messages)
        
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
        return self._fallback_analysis(column_name, sample_data, data_type)
    
    def _fallback_analysis(self, column_name: str, sample_data: List[str], 
                          data_type: str) -> Dict[str, Any]:
        """Basic fallback analysis when AI is unavailable"""
        return {
            "semantic_type": "Unknown",
            "category": "General",
            "country_context": "Global",
            "validation_rules": [],
            "data_quality_checks": ["Completeness", "Uniqueness"],
            "confidence_score": 0
        }
    
    def generate_dynamic_rules(self, df: pd.DataFrame, column_name: str, 
                              profile: Any) -> List[Dict[str, Any]]:
        """Generate comprehensive validation rules for a column using AI"""
        
        sample_data = df[column_name].dropna().astype(str).head(100).tolist()
        data_type = str(profile.dtype)
        
        # Get AI analysis
        ai_analysis = self.analyze_column_semantic_type(column_name, sample_data, data_type)
        
        rules = []
        
        # Generate Column info
        col_info = {
            'Column': column_name,
            'Dimension': 'Metadata',
            'Type': 'AI_CLASSIFICATION',
            'Condition': ai_analysis.get('semantic_type', 'Unknown'),
            'Valid': 'N/A',
            'Invalid': 'N/A',
            'Example_Valid': ai_analysis.get('validation_rules', [{}])[0].get('example_valid', 'N/A') if ai_analysis.get('validation_rules') else 'N/A',
            'Severity': 'INFO',
            'Confidence': ai_analysis.get('confidence_score', 0),
            'Category': ai_analysis.get('category', 'General'),
            'Country': ai_analysis.get('country_context', 'Global')
        }
        rules.append(col_info)
        
        # Process AI-generated validation rules
        for rule in ai_analysis.get('validation_rules', []):
            validation_rule = {
                'Column': column_name,
                'Dimension': 'Validity',
                'Type': rule.get('rule_name', 'VALIDATION'),
                'Condition': rule.get('condition', ''),
                'Valid': rule.get('example_valid', 'N/A'),
                'Invalid': rule.get('example_invalid', 'N/A'),
                'Example_Valid': rule.get('example_valid', 'N/A'),
                'Severity': rule.get('severity', 'MEDIUM'),
                'Condition_Type': rule.get('condition_type', 'custom'),
                'Description': rule.get('description', ''),
                'AI_Generated': True
            }
            rules.append(validation_rule)
        
        # Add standard DQ checks based on profile
        if profile.null_percentage > 0:
            rules.append({
                'Column': column_name,
                'Dimension': 'Completeness',
                'Type': 'NOT_NULL',
                'Condition': f'{column_name} IS NOT NULL',
                'Valid': 'Any non-null value',
                'Invalid': 'NULL/NaN',
                'Example_Valid': str(sample_data[0]) if sample_data else 'N/A',
                'Severity': 'CRITICAL' if profile.null_percentage > 20 else 'HIGH' if profile.null_percentage > 5 else 'MEDIUM',
                'Null_Percentage': profile.null_percentage
            })
        
        # Uniqueness check for near-unique columns
        unique_pct = profile.unique_percentage
        if 95 <= unique_pct < 100:
            dup_count = profile.total_rows - profile.unique_count
            rules.append({
                'Column': column_name,
                'Dimension': 'Uniqueness',
                'Type': 'UNIQUE_CHECK',
                'Condition': f'{column_name} must be unique',
                'Valid': 'Unique values only',
                'Invalid': f'Duplicate values ({dup_count} found)',
                'Example_Valid': 'Single occurrence per value',
                'Severity': 'HIGH',
                'Duplicate_Count': dup_count,
                'Unique_Percentage': unique_pct
            })
        
        # Length validation for text columns
        if 'object' in str(profile.dtype):
            min_len = getattr(profile, 'min_length', 0)
            max_len = getattr(profile, 'max_length', 0)
            if max_len > 0:
                rules.append({
                    'Column': column_name,
                    'Dimension': 'Validity',
                    'Type': 'LENGTH_CHECK',
                    'Condition': f'Length between {min_len} and {max_len} characters',
                    'Valid': f'Text length {min_len}-{max_len}',
                    'Invalid': f'Length outside range',
                    'Example_Valid': f'Mid-range length (~{(min_len+max_len)//2})',
                    'Severity': 'MEDIUM' if (max_len - min_len) > 50 else 'LOW',
                    'Min_Length': min_len,
                    'Max_Length': max_len
                })
        
        return rules
    
    def validate_data_against_rules(self, df: pd.DataFrame, rules: List[Dict]) -> pd.DataFrame:
        """Validate dataframe against AI-generated rules and return results"""
        validation_results = []
        
        for rule in rules:
            if rule.get('Type') in ['AI_CLASSIFICATION', 'INFO']:
                continue
                
            column = rule.get('Column')
            if column not in df.columns:
                continue
            
            condition_type = rule.get('Condition_Type', 'custom')
            condition = rule.get('Condition', '')
            
            valid_count = 0
            invalid_count = 0
            invalid_samples = []
            
            if condition_type == 'regex' and condition:
                try:
                    pattern = re.compile(condition)
                    for idx, value in df[column].dropna().astype(str).items():
                        if pattern.match(value.strip()):
                            valid_count += 1
                        else:
                            invalid_count += 1
                            if len(invalid_samples) < 5:
                                invalid_samples.append(value[:50])
                except re.error:
                    pass
            
            elif rule.get('Type') == 'NOT_NULL':
                valid_count = df[column].notna().sum()
                invalid_count = df[column].isna().sum()
            
            elif rule.get('Type') == 'UNIQUE_CHECK':
                value_counts = df[column].value_counts()
                valid_count = (value_counts == 1).sum()
                invalid_count = (value_counts > 1).sum()
            
            elif rule.get('Type') == 'LENGTH_CHECK':
                min_len = rule.get('Min_Length', 0)
                max_len = rule.get('Max_Length', 999999)
                lengths = df[column].dropna().astype(str).str.len()
                valid_count = ((lengths >= min_len) & (lengths <= max_len)).sum()
                invalid_count = ((lengths < min_len) | (lengths > max_len)).sum()
            
            total = valid_count + invalid_count
            coverage = (valid_count / total * 100) if total > 0 else 0
            
            validation_results.append({
                'Column': column,
                'Rule_Type': rule.get('Type'),
                'Dimension': rule.get('Dimension'),
                'Valid_Count': valid_count,
                'Invalid_Count': invalid_count,
                'Coverage_Percent': round(coverage, 2),
                'Invalid_Samples': invalid_samples,
                'Severity': rule.get('Severity', 'MEDIUM'),
                'Condition': condition
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
        
        # Create comprehensive output
        output_data = []
        for rule in all_rules:
            # Find matching validation result
            match = validation_results[
                (validation_results['Column'] == rule.get('Column')) & 
                (validation_results['Rule_Type'] == rule.get('Type'))
            ]
            
            row = {
                'Column': rule.get('Column'),
                'Dimension': rule.get('Dimension'),
                'Type': rule.get('Type'),
                'Condition': rule.get('Condition'),
                'Valid': rule.get('Valid', 'N/A'),
                'Invalid': rule.get('Invalid', 'N/A'),
                'Example_Valid': rule.get('Example_Valid', 'N/A'),
                'Severity': rule.get('Severity', 'MEDIUM'),
                'AI_Generated': rule.get('AI_Generated', False),
                'Category': rule.get('Category', 'General'),
                'Country': rule.get('Country', 'Global'),
                'Confidence': rule.get('Confidence', 0)
            }
            
            if not match.empty:
                row['Valid_Count'] = match.iloc[0]['Valid_Count']
                row['Invalid_Count'] = match.iloc[0]['Invalid_Count']
                row['Coverage_Percent'] = match.iloc[0]['Coverage_Percent']
                row['Invalid_Samples'] = ', '.join(match.iloc[0]['Invalid_Samples'][:3])
            else:
                row['Valid_Count'] = 'N/A'
                row['Invalid_Count'] = 'N/A'
                row['Coverage_Percent'] = 'N/A'
                row['Invalid_Samples'] = 'N/A'
            
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

    st.set_page_config(page_title="AI Data Profiler", layout="wide")

    # Executive Dashboard
    _render_executive_dashboard()

    # Main Tabs
    tabs = st.tabs([
        "📊 Overview", 
        "📋 Column Profiles", 
        "🤖 AI Validations", 
        "🎯 Match Rules", 
        "⚠️ Quality Issues", 
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
        _render_issues_tab()
    with tabs[5]: 
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

            # Duplicate analysis
            if prof.unique_count < prof.total_rows:
                dups = find_duplicate_groups(df, col)
                if dups:
                    with st.expander(f"⚠️ Duplicates ({len(dups)} groups)"):
                        for d in dups[:5]:
                            st.write(f"- Value '{d['value'][:30]}' appears {d['count']} times ({d['percentage']}%)")


def _render_ai_validations_tab():
    """NEW: AI-Powered Validations Tab"""
    state = st.session_state.app_state
    df = state.df
    profiles = state.column_profiles
    
    st.subheader("🤖 AI-Powered Data Validation Rules")
    
    # Check Azure OpenAI configuration
    missing_config = AzureOpenAIConfig.validate()
    if missing_config:
        st.error(f"⚠️ Azure OpenAI not configured. Missing: {', '.join(missing_config)}")
        st.info("Please set these in your Streamlit secrets or environment variables.")
        return
    
    if st.button("🚀 Generate AI Validation Rules", type="primary"):
        with st.spinner("🤖 Analyzing data with Azure OpenAI..."):
            detector = DynamicValidationDetector()
            validation_df = detector.generate_comprehensive_dq_rules(df, profiles)
            
            # Store in session state
            state.ai_validation_rules = validation_df
            
            st.success(f"✅ Generated {len(validation_df)} validation rules!")
    
    if hasattr(state, 'ai_validation_rules') and not state.ai_validation_rules.empty:
        validation_df = state.ai_validation_rules
        
        # Summary metrics
        ai_generated = validation_df[validation_df['AI_Generated'] == True]
        st.metric("AI-Generated Rules", len(ai_generated))
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            dim_filter = st.multiselect("Dimension", validation_df['Dimension'].unique())
        with col2:
            sev_filter = st.multiselect("Severity", validation_df['Severity'].unique())
        with col3:
            cat_filter = st.multiselect("Category", validation_df['Category'].unique())
        
        # Apply filters
        filtered_df = validation_df.copy()
        if dim_filter:
            filtered_df = filtered_df[filtered_df['Dimension'].isin(dim_filter)]
        if sev_filter:
            filtered_df = filtered_df[filtered_df['Severity'].isin(sev_filter)]
        if cat_filter:
            filtered_df = filtered_df[filtered_df['Category'].isin(cat_filter)]
        
        # Display validation rules table
        st.subheader("Validation Rules")
        
        # Reorder columns for better display
        display_cols = [
            'Column', 'Dimension', 'Type', 'Condition', 'Valid', 'Invalid', 
            'Example_Valid', 'Severity', 'AI_Generated', 'Category', 'Country',
            'Valid_Count', 'Invalid_Count', 'Coverage_Percent'
        ]
        
        # Only include columns that exist
        display_cols = [c for c in display_cols if c in filtered_df.columns]
        
        # Color coding for severity
        def color_severity(val):
            if isinstance(val, str):
                colors = {
                    'CRITICAL': 'background-color: #dc2626; color: white',
                    'HIGH': 'background-color: #ea580c; color: white',
                    'MEDIUM': 'background-color: #ca8a04; color: white',
                    'LOW': 'background-color: #16a34a; color: white'
                }
                return colors.get(val, '')
            return ''
        
        styled_df = filtered_df[display_cols].style.applymap(
            color_severity, subset=['Severity'] if 'Severity' in display_cols else []
        )
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Visualization of validation coverage
        st.subheader("Validation Coverage Analysis")
        
        if 'Coverage_Percent' in filtered_df.columns:
            # FIX: Convert Coverage_Percent to numeric, coercing errors to NaN
            coverage_data = filtered_df.copy()
            coverage_data['Coverage_Percent_Numeric'] = pd.to_numeric(
                coverage_data['Coverage_Percent'], 
                errors='coerce'
            )
            
            # Filter out rows where Coverage_Percent could not be converted to a number
            coverage_data = coverage_data.dropna(subset=['Coverage_Percent_Numeric'])
            
            if not coverage_data.empty:
                # FIX: Ensure Valid_Count and Invalid_Count are also numeric for hover_data
                for col in ['Valid_Count', 'Invalid_Count']:
                    if col in coverage_data.columns:
                        coverage_data[col] = pd.to_numeric(coverage_data[col], errors='coerce')
                
                fig = px.scatter(
                    coverage_data, 
                    x='Column', 
                    y='Type',
                    size='Coverage_Percent_Numeric',  # Use the numeric column
                    color='Severity',
                    hover_data=['Valid_Count', 'Invalid_Count', 'Coverage_Percent'],
                    title="Validation Coverage by Column and Rule Type",
                    size_max=50  # Limit max bubble size
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric coverage data available for visualization")
        
        # AI Confidence Distribution
        if 'Confidence' in filtered_df.columns:
            conf_data = filtered_df[filtered_df['AI_Generated'] == True].copy()
            
            # FIX: Convert Confidence to numeric
            conf_data['Confidence_Numeric'] = pd.to_numeric(
                conf_data['Confidence'], 
                errors='coerce'
            )
            conf_data = conf_data.dropna(subset=['Confidence_Numeric'])
            
            if not conf_data.empty:
                fig = px.histogram(
                    conf_data, 
                    x='Confidence_Numeric',  # Use numeric column
                    nbins=10,
                    title="AI Confidence Score Distribution",
                    color='Category',
                    labels={'Confidence_Numeric': 'Confidence Score'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric confidence data available for visualization")

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


def _render_issues_tab():
    state = st.session_state.app_state

    st.subheader("⚠️ Data Quality Issues")

    # Use AI-generated rules if available, otherwise generate on the fly
    if hasattr(state, 'ai_validation_rules') and not state.ai_validation_rules.empty:
        dq_rules = state.ai_validation_rules[
            state.ai_validation_rules['Dimension'] != 'Metadata'
        ].to_dict('records')
    else:
        st.info("Generate AI validation rules first to see detailed quality issues")
        return

    if not dq_rules:
        st.success("✅ No quality issues detected!")
        return

    # Severity distribution
    sev_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for r in dq_rules:
        sev = r.get('Severity', 'MEDIUM')
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    col1, col2 = st.columns([1, 2])

    with col1:
        fig = px.pie(
            values=list(sev_counts.values()), 
            names=list(sev_counts.keys()),
            title="Issues by Severity",
            color=list(sev_counts.keys()),
            color_discrete_map={
                'CRITICAL': '#dc2626', 
                'HIGH': '#ea580c', 
                'MEDIUM': '#ca8a04', 
                'LOW': '#16a34a'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Issues by dimension
        dim_counts = {}
        for r in dq_rules:
            dim = r.get('Dimension', 'Other')
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        fig = px.bar(
            x=list(dim_counts.keys()), 
            y=list(dim_counts.values()),
            title="Issues by DQ Dimension", 
            color=list(dim_counts.keys())
        )
        st.plotly_chart(fig, use_container_width=True)

    # Issues table
    st.subheader("Quality Rules & Actions")
    issues_df = pd.DataFrame(dq_rules)

    def color_severity(val):
        colors = {
            'CRITICAL': '#dc2626', 
            'HIGH': '#ea580c', 
            'MEDIUM': '#ca8a04', 
            'LOW': '#16a34a'
        }
        return f'background-color: {colors.get(val, "gray")}; color: white'

    display_cols = ['Column', 'Dimension', 'Type', 'Severity', 'Valid', 'Invalid', 'Example_Valid']
    display_cols = [c for c in display_cols if c in issues_df.columns]
    
    styled = issues_df[display_cols].style.applymap(
        color_severity, 
        subset=['Severity'] if 'Severity' in display_cols else []
    )
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

            # 4. AI Validation Rules (NEW - Replaces Pattern Found)
            progress.progress(55)
            if hasattr(state, 'ai_validation_rules') and not state.ai_validation_rules.empty:
                # Reorder columns for better readability
                validation_df = state.ai_validation_rules.copy()
                col_order = [
                    'Column', 'Dimension', 'Type', 'Condition', 'Valid', 'Invalid',
                    'Example_Valid', 'Severity', 'Category', 'Country', 'AI_Generated',
                    'Confidence', 'Valid_Count', 'Invalid_Count', 'Coverage_Percent'
                ]
                # Only include columns that exist
                col_order = [c for c in col_order if c in validation_df.columns]
                validation_df = validation_df[col_order]
                validation_df.to_excel(writer, sheet_name='AI Validation Rules', index=False)
            else:
                pd.DataFrame({'Message': ['No AI validation rules generated. Run validation analysis first.']}).to_excel(
                    writer, sheet_name='AI Validation Rules', index=False)

            # 5. Match Rule Suggestion
            progress.progress(75)
            match_rules = generate_match_rules(df, profiles)
            pd.DataFrame(match_rules).to_excel(writer, sheet_name='Match Rule Suggestion', index=False)

            # 6. Data Quality Summary
            progress.progress(90)
            if hasattr(state, 'ai_validation_rules') and not state.ai_validation_rules.empty:
                # Create summary by dimension and severity
                dq_summary = state.ai_validation_rules.groupby(
                    ['Dimension', 'Severity']
                ).size().reset_index(name='Rule Count')
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
                'ai_powered': True
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
            'ai_validation_rules': state.ai_validation_rules.to_dict('records') if hasattr(state, 'ai_validation_rules') else [],
            'match_rules': generate_match_rules(df, profiles)
        }

        st.download_button(
            "⬇️ Download JSON Report", 
            data=json.dumps(report, indent=2, default=str),
            file_name=f"ai_data_profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        show_toast("JSON report generated!", "success")
    except Exception as e:
        st.error(f"Error generating JSON: {str(e)}")