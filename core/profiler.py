"""Core profiling engine"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from models import ColumnProfile, DataQualityReport, DuplicateGroup
from utils.fuzzy_matching import FuzzyMatcher
from utils.data_utils import detect_special_characters, find_exact_duplicates, generate_column_suggestions
from utils.concurrency import ParallelProcessor, TimeEstimator
from utils.dtype_mapper import get_human_readable_dtype
import re


class DataProfilerEngine:
    def __init__(self, df: pd.DataFrame, filename: str = "data"):
        self.df = df.copy()
        self.original_df = df.copy()
        self.filename = filename
        self.column_profiles: Dict[str, ColumnProfile] = {}
        self.quality_report: Optional[DataQualityReport] = None
        self.exact_duplicates: List[DuplicateGroup] = []
        self.fuzzy_duplicates: List[DuplicateGroup] = []
        
    def analyze_column(self, column: str) -> ColumnProfile:
        series = self.df[column]
        total_rows = len(series)
        
        # Volume & Completeness
        null_count = series.isnull().sum()
        null_percentage = (null_count / total_rows) * 100 if total_rows > 0 else 0
        non_null_count = total_rows - null_count
        non_null_percentage = 100 - null_percentage
        
        # Uniqueness
        unique_count = series.nunique()
        unique_percentage = (unique_count / total_rows) * 100 if total_rows > 0 else 0
        duplicate_count = total_rows - unique_count
        duplicate_percentage = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
        
        memory_bytes = series.memory_usage(deep=True)
        memory_usage = f"{memory_bytes / 1024:.2f} KB" if memory_bytes > 1024 else f"{memory_bytes} bytes"
        
        # Enhanced special character detection
        special_chars = detect_special_characters(series)
        
        # Calculate total rows with special chars (approximate if expensive)
        total_special_char_rows = 0
        if (pd.api.types.is_string_dtype(series) or series.dtype == 'object') and len(special_chars) > 0:
            # Simple check for any special char from our list
            # Constructing a regex from found special chars for efficiency
            found_symbols = [re.escape(c['symbol']) for c in special_chars if 'symbol' in c]
            if found_symbols:
                pattern = '|'.join(found_symbols)
                try:
                    total_special_char_rows = int(series.dropna().astype(str).str.contains(pattern, regex=True).sum())
                except:
                    pass
        
        # Data pattern analysis
        patterns = self._analyze_patterns(series)
        pattern_counts = {} # _analyze_patterns could be updated to return this, or we estimate
        if patterns.get('pattern_types'):
            # For now, just mark presence. 
            # Real counting of patterns across full dataset is expensive.
            for p_type in patterns['pattern_types']:
                pattern_counts[p_type] = -1 # Indicates "Present" without exact count
        
        # Accuracy and validity checks
        accuracy_info = self._check_accuracy(series)
        validity_info = self._check_validity(series)
        
        # Completeness: blank vs null
        blank_count = self._count_blanks(series)
        
        # Formatting info
        formatting_info = self._analyze_formatting(series)
        
        # Length analysis
        length_stats = self._analyze_lengths(series)
        
        # Business Rules
        business_rules = self._check_business_rules(series, str(series.dtype))
        
        # Outlier detection
        outlier_info = {"count": 0, "bounds": (None, None), "samples": []}
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            try:
                # Ensure series is float for calculations
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if len(numeric_series) > 0:
                    Q1 = numeric_series.quantile(0.25)
                    Q3 = numeric_series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
                    outlier_info = {
                        "count": len(outliers),
                        "bounds": (float(lower_bound), float(upper_bound)),
                        "samples": outliers.head(3).tolist()
                    }
            except Exception as e:
                pass
        
        sample_values = series.dropna().head(3).tolist()
        
        fuzzy_keys = []
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            if unique_count / total_rows < 0.9:
                fuzzy_keys.append("High similarity potential")
        
        suggestions = generate_column_suggestions(column, series, null_percentage, 
                                                unique_percentage, special_chars, 
                                                outlier_info, fuzzy_keys, blank_count)
                                                
        # Prepare profile data for Risk Assessment & Cleansing
        profile_dict = {
            'total_rows': total_rows,
            'null_count': null_count,
            'null_percentage': null_percentage,
            'unique_count': unique_count,
            'duplicate_percentage': duplicate_percentage,
            'special_char_rows': total_special_char_rows,
            'outliers': outlier_info,
            'formatting': formatting_info
        }
        
        risk = self._assess_risk(profile_dict)
        cleansing = self._generate_cleansing_recommendations(profile_dict)
        
        return ColumnProfile(
            column_name=column,
            dtype=str(series.dtype),
            human_readable_dtype=get_human_readable_dtype(series.dtype),
            total_rows=total_rows,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            duplicate_count=duplicate_count,
            memory_usage=memory_usage,
            special_chars=special_chars,
            outliers=outlier_info,
            suggestions=suggestions,
            sample_values=sample_values,
            fuzzy_keys=fuzzy_keys,
            patterns=patterns,
            accuracy_info=accuracy_info,
            validity_info=validity_info,
            blank_count=blank_count,
            formatting_info=formatting_info,
            # New enhanced fields
            non_null_count=non_null_count,
            non_null_percentage=non_null_percentage,
            duplicate_percentage=duplicate_percentage,
            min_length=length_stats['min'],
            max_length=length_stats['max'],
            avg_length=length_stats['avg'],
            out_of_bounds_count=length_stats['out_of_bounds'],
            pattern_counts=pattern_counts,
            total_special_char_rows=total_special_char_rows,
            business_rule_violations=business_rules,
            risk_score=risk['score'],
            risk_level=risk['level'],
            key_issues=risk['key_issues'],
            cleansing_recommendations=cleansing
        )
    
    def _analyze_patterns(self, series: pd.Series) -> Dict:
        """Analyze data patterns in the column using optimized unique value checks"""
        patterns = {
            'pattern_types': [],
            'examples': {}
        }
        
        try:
            if pd.api.types.is_string_dtype(series):
                # Work with unique non-null values for performance
                unique_values = series.dropna().unique()
                if len(unique_values) == 0:
                    return patterns
                    
                # Limit to first 1000 unique values for pattern detection to prevent OOM on high cardinality
                sample_source = pd.Series(unique_values[:1000]).astype(str)
                
                # Check for email pattern
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if sample_source.str.match(email_pattern).any():
                    patterns['pattern_types'].append('Email')
                    # Find an example
                    examples = sample_source[sample_source.str.match(email_pattern)]
                    if not examples.empty:
                        patterns['examples']['Email'] = examples.iloc[0]
                
                # Check for phone pattern
                phone_pattern = r'^[\d\s\-\+\(\)]{7,20}$'
                if sample_source.str.match(phone_pattern).any():
                     # Distinguish from simple numbers
                    if not sample_source.str.isnumeric().all():
                        patterns['pattern_types'].append('Phone')
                        examples = sample_source[sample_source.str.match(phone_pattern)]
                        if not examples.empty:
                            patterns['examples']['Phone'] = examples.iloc[0]
                
                # Check for date pattern
                date_patterns = [
                    r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # DD/MM/YYYY
                    r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',    # YYYY/MM/DD
                ]
                for dp in date_patterns:
                    if sample_source.str.match(dp).any():
                        patterns['pattern_types'].append('Date')
                        examples = sample_source[sample_source.str.match(dp)]
                        if not examples.empty:
                            patterns['examples']['Date'] = examples.iloc[0]
                        break
                
                # Check for URL pattern
                url_pattern = r'^https?://[^\s<>\"{}|\\^`\[\]]+$'
                if sample_source.str.match(url_pattern).any():
                    patterns['pattern_types'].append('URL')
                    examples = sample_source[sample_source.str.match(url_pattern)]
                    if not examples.empty:
                        patterns['examples']['URL'] = examples.iloc[0]
                
                # Length statistics - efficient calculation on full series
                # We can approximate average length from uniques if series is huge, but str.len() is usually fast enough
                # For very large series, maybe skip or sample
                if len(series) > 100000:
                    lengths = series.head(10000).astype(str).str.len()
                else:
                    lengths = series.dropna().astype(str).str.len()
                    
                if not lengths.empty:
                    patterns['length_stats'] = {
                        'min': int(lengths.min()),
                        'max': int(lengths.max()),
                        'avg': round(lengths.mean(), 2)
                    }
            
            elif pd.api.types.is_numeric_dtype(series):
                patterns['pattern_types'].append('Numeric')
                non_null = series.dropna()
                if not non_null.empty:
                    patterns['range'] = {
                        'min': float(non_null.min()),
                        'max': float(non_null.max())
                    }
                    patterns['negative_count'] = int((non_null < 0).sum())
                    patterns['zero_count'] = int((non_null == 0).sum())
            
            elif pd.api.types.is_datetime64_any_dtype(series):
                patterns['pattern_types'].append('DateTime')
                non_null = series.dropna()
                if not non_null.empty:
                    patterns['range'] = {
                        'min': str(non_null.min()),
                        'max': str(non_null.max())
                    }
                    
        except Exception:
            # Fail gracefully on pattern detection
            pass
            
        return patterns
    
    def _check_accuracy(self, series: pd.Series) -> Dict:
        """Check data accuracy"""
        accuracy = {
            'score': 100,
            'issues': []
        }
        
        if pd.api.types.is_string_dtype(series):
            # Check for inconsistent spacing
            sample = series.dropna().astype(str)
            inconsistent_spaces = sample[sample.str.contains(r'\s{2,}', regex=True, na=False)]
            if len(inconsistent_spaces) > 0:
                accuracy['issues'].append(f"{len(inconsistent_spaces)} values with multiple spaces")
                accuracy['score'] -= min(10, len(inconsistent_spaces) / len(sample) * 100)
            
            # Check for leading/trailing spaces
            leading_trailing = sample[sample != sample.str.strip()]
            if len(leading_trailing) > 0:
                accuracy['issues'].append(f"{len(leading_trailing)} values with leading/trailing spaces")
                accuracy['score'] -= min(15, len(leading_trailing) / len(sample) * 100)
        
        accuracy['score'] = max(0, min(100, accuracy['score']))
        return accuracy
    
    def _check_validity(self, series: pd.Series) -> Dict:
        """Check data validity"""
        validity = {
            'score': 100,
            'invalid_count': 0,
            'invalid_examples': []
        }
        
        if pd.api.types.is_numeric_dtype(series):
            # Check for infinity values
            inf_count = np.isinf(series).sum()
            if inf_count > 0:
                validity['invalid_count'] += inf_count
                validity['invalid_examples'].append(f"{inf_count} infinite values")
                validity['score'] -= min(20, inf_count / len(series) * 100)
        
        elif pd.api.types.is_string_dtype(series):
            # Check for empty strings
            empty_strings = (series == '').sum()
            if empty_strings > 0:
                validity['invalid_count'] += empty_strings
                validity['invalid_examples'].append(f"{empty_strings} empty strings")
                validity['score'] -= min(10, empty_strings / len(series) * 100)
        
        validity['score'] = max(0, min(100, validity['score']))
        return validity
    
    def _count_blanks(self, series: pd.Series) -> Dict:
        """Count blank vs null values"""
        blanks = {
            'null_count': int(series.isnull().sum()),
            'empty_string_count': 0,
            'whitespace_only_count': 0
        }
        
        if pd.api.types.is_string_dtype(series):
            str_series = series.astype(str)
            blanks['empty_string_count'] = int((str_series == '').sum())
            blanks['whitespace_only_count'] = int(str_series.str.match(r'^\s+$', na=False).sum())
        
        blanks['total_missing'] = blanks['null_count'] + blanks['empty_string_count'] + blanks['whitespace_only_count']
        return blanks
    
    def _analyze_formatting(self, series: pd.Series) -> Dict:
        """Analyze formatting consistency"""
        formatting = {
            'consistent_case': True,
            'case_type': None,
            'format_examples': {}
        }
        
        if pd.api.types.is_string_dtype(series):
            sample = series.dropna().astype(str)
            if len(sample) > 0:
                upper_count = sample.str.isupper().sum()
                lower_count = sample.str.islower().sum()
                title_count = sample.str.istitle().sum()
                total = len(sample)
                
                if upper_count / total > 0.8:
                    formatting['case_type'] = 'UPPER'
                elif lower_count / total > 0.8:
                    formatting['case_type'] = 'lower'
                elif title_count / total > 0.8:
                    formatting['case_type'] = 'Title'
                else:
                    formatting['consistent_case'] = False
                    formatting['case_type'] = 'Mixed'
                
                formatting['case_distribution'] = {
                    'upper': int(upper_count),
                    'lower': int(lower_count),
                    'title': int(title_count),
                    'mixed': int(total - upper_count - lower_count - title_count)
                }
        
        return formatting

    def _analyze_lengths(self, series: pd.Series) -> Dict:
        """Analyze string lengths"""
        length_stats = {
            'min': 0,
            'max': 0,
            'avg': 0.0,
            'out_of_bounds': 0
        }
        
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            # Efficiently calculate lengths
            # Convert to string and measure length
            # To avoid OOM on huge datasets, we might want to sample if very large
            if len(series) > 1000000:
                sample = series.dropna().sample(100000)
                lengths = sample.astype(str).str.len()
            else:
                lengths = series.dropna().astype(str).str.len()
            
            if len(lengths) > 0:
                length_stats['min'] = int(lengths.min())
                length_stats['max'] = int(lengths.max())
                length_stats['avg'] = float(lengths.mean())
                
                # Assume "out of bounds" is anything > 255 chars (common DB limit) or 0
                # This is a heuristic
                out_bounds = (lengths > 255) | (lengths == 0)
                length_stats['out_of_bounds'] = int(out_bounds.sum())
                
        return length_stats

    def _check_business_rules(self, series: pd.Series, dtype: str) -> List[str]:
        """Check standard business rules"""
        violations = []
        
        try:
            non_null = series.dropna()
            total = len(non_null)
            if total == 0:
                return violations
            
            # 1. Negative values for counts/amounts (heuristic)
            if pd.api.types.is_numeric_dtype(series):
                negatives = (non_null < 0).sum()
                if negatives > 0:
                    violations.append(f"Negative values detected ({negatives} rows) - Verify if allowed")
            
            # 2. Future dates for birthdates or past events (heuristic)
            if pd.api.types.is_datetime64_any_dtype(series):
                future = (non_null > pd.Timestamp.now()).sum()
                if future > 0:
                    violations.append(f"Future dates detected ({future} rows) - Verify validity")
            
            # 3. Email format
            if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
                # Simple check if column name looks like email
                # We can't know column name here easily without passing it, but we can check content signature
                sample = non_null.head(100).astype(str)
                email_matches = sample.str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$').sum()
                if email_matches > 50: # Likely an email column
                    # Check full series for invalid formats
                    invalid_emails = (~non_null.astype(str).str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$')).sum()
                    if invalid_emails > 0:
                         violations.append(f"Invalid email formats detected ({invalid_emails} rows)")
        
        except Exception:
            pass
            
        return violations

    def _assess_risk(self, profile_data: Dict) -> Dict:
        """Assess data quality risk"""
        risk = {
            'score': 0,
            'level': 'Low',
            'key_issues': []
        }
        
        score = 0
        issues = []
        
        # Nulls
        if profile_data['null_percentage'] > 50:
            score += 40
            issues.append("Critical: Majority of data is missing")
        elif profile_data['null_percentage'] > 20:
            score += 20
            issues.append("High missing value rate")
        
        # Duplicates
        if profile_data.get('duplicate_percentage', 0) > 50:
             # Only a risk if not a categorical/flag column
             if profile_data['unique_count'] > 5: 
                score += 30
                issues.append("High duplication rate in potential identifier")
        
        # Special characters
        if profile_data.get('special_char_rows', 0) > 0:
            # Impact depends on percentage
            pct = (profile_data['special_char_rows'] / profile_data['total_rows']) * 100
            if pct > 10:
                score += 15
                issues.append("Frequent special characters - potential ingestion risk")
        
        # Outliers
        if profile_data.get('outliers', {}).get('count', 0) > 0:
            score += 10
            issues.append("Statistical outliers detected")
            
        risk['score'] = min(100, score)
        if score >= 60:
            risk['level'] = 'High'
        elif score >= 30:
            risk['level'] = 'Medium'
        else:
            risk['level'] = 'Low'
            
        risk['key_issues'] = issues
        return risk

    def _generate_cleansing_recommendations(self, profile_data: Dict) -> List[Dict]:
        """Generate actionable cleansing steps"""
        recommendations = []
        
        # Missing values
        null_pct = profile_data['null_percentage']
        if null_pct > 0:
            if null_pct < 5:
                # Few missing - drop?
                recommendations.append({
                    'action': 'Drop Rows',
                    'impact': f"{profile_data['null_count']} rows",
                    'description': 'Remove rows with missing values (low impact)'
                })
                # Or fill?
                recommendations.append({
                    'action': 'Impute Values',
                    'impact': f"{profile_data['null_count']} rows",
                    'description': 'Fill missing values with mode/mean or "Unknown"'
                })
            else:
                 recommendations.append({
                    'action': 'Impute/Flag',
                    'impact': f"{profile_data['null_count']} rows",
                    'description': 'High missing rate - investigate source or flag as missing'
                })
        
        # Special characters
        if profile_data.get('special_char_rows', 0) > 0:
             recommendations.append({
                'action': 'Clean Strings',
                'impact': f"{profile_data['special_char_rows']} rows",
                'description': 'Remove or replace detected special characters using regex'
            })
            
        # Spacing
        # We can implement specific checks for leading/trailing if we want detailed recs
        
        # Case
        formatting = profile_data.get('formatting', {})
        if not formatting.get('consistent_case', True):
             recommendations.append({
                'action': 'Standardize Case',
                'impact': 'All rows',
                'description': 'Convert text to Title Case or Upper Case for consistency'
            })
            
        return recommendations
    
    def find_exact_duplicates(self, subset: Optional[List[str]] = None) -> List[DuplicateGroup]:
        self.exact_duplicates = find_exact_duplicates(self.df, subset)
        return self.exact_duplicates
    
    def find_fuzzy_duplicates(self, columns: List[str], threshold: float = 85.0, algorithm: str = 'rapidfuzz') -> List[DuplicateGroup]:
        matcher = FuzzyMatcher(algorithm=algorithm, threshold=threshold)
        all_groups = []
        group_id = 0
        
        for col in columns:
            if not pd.api.types.is_string_dtype(self.df[col]):
                continue
            
            col_groups = matcher.find_duplicate_groups(self.df[col])
            for group in col_groups:
                group_id += 1
                group.group_id = group_id
                group.values = [self.df.loc[idx].to_dict() for idx in group.indices]
                all_groups.append(group)
        
        self.fuzzy_duplicates = all_groups
        return all_groups
    
    def find_combined_duplicates(self, exact_columns: List[str], fuzzy_columns: List[str], 
                                threshold: float = 85.0, algorithm: str = 'rapidfuzz') -> List[DuplicateGroup]:
        """Find duplicates using both exact and fuzzy matching on different columns"""
        groups = []
        
        # First, group by exact columns
        if exact_columns:
            exact_groups = self.find_exact_duplicates(subset=exact_columns)
            
            # Within each exact group, check fuzzy matching on fuzzy columns
            if fuzzy_columns and exact_groups:
                matcher = FuzzyMatcher(algorithm=algorithm, threshold=threshold)
                
                for exact_group in exact_groups:
                    indices = exact_group.indices
                    if len(indices) < 2:
                        continue
                    
                    # Check fuzzy similarity within this exact group
                    for fuzzy_col in fuzzy_columns:
                        if not pd.api.types.is_string_dtype(self.df[fuzzy_col]):
                            continue
                        
                        # Get values for this column within the exact group
                        group_df = self.df.loc[indices]
                        col_groups = matcher.find_duplicate_groups(group_df[fuzzy_col])
                        
                        for col_group in col_groups:
                            if len(col_group.indices) > 1:
                                # The col_group.indices are already the original DataFrame indices
                                # because we passed a Series with the original index preserved
                                original_indices = col_group.indices
                                groups.append(DuplicateGroup(
                                    group_id=len(groups) + 1,
                                    indices=original_indices,
                                    values=[self.df.loc[idx].to_dict() for idx in original_indices],
                                    match_type='combined',
                                    similarity_score=col_group.similarity_score,
                                    key_columns=exact_columns + [fuzzy_col],
                                    representative_value=str(self.df.loc[original_indices[0], fuzzy_col])
                                ))
        
        # Also find pure fuzzy duplicates if no exact columns specified
        if not exact_columns and fuzzy_columns:
            return self.find_fuzzy_duplicates(fuzzy_columns, threshold, algorithm)
        
        return groups
    
    def profile(self, fast_mode: bool = False, progress_callback=None):
        total_columns = len(self.df.columns)
        
        # Initialize time estimator
        timer = TimeEstimator(total_columns)
        
        def update_progress_wrapper(current_idx, col_name):
            if progress_callback:
                try:
                    metrics = timer.get_metrics(current_idx)
                    progress_callback({
                        'message': f"Analyzing {col_name}",
                        'percent': int((current_idx / total_columns) * 90), # 0-90% for profiling
                        'elapsed': metrics['elapsed'],
                        'eta': metrics['eta'],
                        'rate': metrics['rate'],
                        'current': current_idx,
                        'total': total_columns
                    })
                except Exception:
                    pass

        # Use ParallelProcessor for column analysis
        with ParallelProcessor() as processor:
            # We need to wrap analyze_column to handle the instance method call correctly
            # and we want to capture the column name for the result
            
            def analyze_wrapper(col):
                # Small update before starting work on this column
                # Note: we can't easily update progress *before* the work starts in parallel map
                # without more complex coordination, so we rely on the map's callback or
                # just update after completion.
                # However, the ParallelProcessor.map doesn't give us per-item start hooks easily.
                # We'll just run the analysis.
                return col, self.analyze_column(col)

            # Define a progress callback for the parallel processor
            def parallel_progress_callback(progress):
                if progress_callback:
                    # Map the processor's 0-100% to our 0-90% range
                    # And inject time metrics
                    # The processor's progress['percent'] is 0-100 based on items completed
                    completed = int(progress['percent'] / 100 * total_columns)
                    metrics = timer.get_metrics(completed)
                    
                    progress_callback({
                        'message': progress['message'],
                        'percent': int((completed / total_columns) * 90),
                        'elapsed': metrics['elapsed'],
                        'eta': metrics['eta'],
                        'rate': metrics['rate'],
                        'current': completed,
                        'total': total_columns
                    })
            
            # Run parallel analysis
            results = processor.map(
                analyze_wrapper, 
                self.df.columns.tolist(),
                progress_callback=parallel_progress_callback if progress_callback else None
            )
            
            # Store results
            for col, profile in results:
                self.column_profiles[col] = profile
            
        if progress_callback:
            try:
                metrics = timer.get_metrics(total_columns)
                progress_callback({
                    'message': "Generating quality report...",
                    'percent': 90,
                    'elapsed': metrics['elapsed'],
                    'eta': "0s",
                    'rate': metrics['rate']
                })
            except Exception:
                pass
                
        self.quality_report = self.generate_quality_report(fast_mode=fast_mode)
    
    def generate_quality_report(self, fast_mode: bool = False):
        total_rows = len(self.df)
        total_columns = len(self.df.columns)
        total_cells = total_rows * total_columns
        missing_cells = self.df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Determine if we should skip expensive checks
        # Skip if fast_mode OR if we already found duplicates elsewhere
        skip_expensive = fast_mode or total_rows > 100000
        
        if not skip_expensive:
            # Only run if not already populated
            if not self.exact_duplicates:
                exact_dups = self.find_exact_duplicates()
            else:
                exact_dups = self.exact_duplicates
                
            exact_duplicate_rows = sum(len(g.indices) - 1 for g in exact_dups)
            
            # Skip auto-fuzzy matching for now, it's too slow for the default report
            # The user can trigger it manually in the "Find Duplicates" tab
            fuzzy_dups = [] 
            fuzzy_duplicate_rows = 0
            fuzzy_groups_count = 0
            
            exact_dup_pct = (exact_duplicate_rows / max(total_rows, 1)) * 100
        else:
            # Fast mode placeholders
            exact_dups = []
            exact_duplicate_rows = 0
            exact_dup_pct = 0
            fuzzy_dups = []
            fuzzy_duplicate_rows = 0
            fuzzy_groups_count = 0
        
        columns_with_issues = [
            col for col, profile in self.column_profiles.items()
            if profile.null_count > 0 or profile.special_chars or (profile.outliers and profile.outliers["count"] > 0)
        ]
        
        score = 100
        score -= min(missing_percentage * 1.5, 25)
        
        if not skip_expensive:
            score -= min((exact_duplicate_rows / max(total_rows, 1)) * 100 * 0.5, 20)
        
        score -= min(len(columns_with_issues) / max(total_columns, 1) * 20, 20)
        
        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_percentage=missing_percentage,
            exact_duplicate_rows=exact_duplicate_rows,
            exact_duplicate_percentage=exact_dup_pct,
            fuzzy_duplicate_groups=fuzzy_groups_count,
            fuzzy_duplicate_rows=fuzzy_duplicate_rows,
            columns_with_issues=columns_with_issues,
            overall_score=max(0, min(100, score)),
            fuzzy_match_summary={
                'groups_found': fuzzy_groups_count,
                'rows_affected': fuzzy_duplicate_rows,
                'scan_columns': []
            }
        )