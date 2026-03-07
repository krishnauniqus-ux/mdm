"""Data models for Advanced Data Profiler Pro"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ColumnProfile:
    column_name: str
    dtype: str
    total_rows: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    duplicate_count: int
    memory_usage: str
    special_chars: List[str]
    outliers: Dict
    suggestions: List[str]
    sample_values: List[Any]
    # Business-friendly data type label
    human_readable_dtype: str = "Unknown"
    # Volume & Completeness
    non_null_count: int = 0
    non_null_percentage: float = 0.0
    
    # Uniqueness & Duplication
    duplicate_percentage: float = 0.0
    
    # Data Type & Length Analysis
    min_length: int = 0
    max_length: int = 0
    avg_length: float = 0.0
    out_of_bounds_count: int = 0
    
    # Patterns & format
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    
    # Special Character Analysis
    total_special_char_rows: int = 0
    
    # Business Rules
    business_rule_violations: List[str] = field(default_factory=list)
    
    # Risk Assessment
    risk_score: int = 0
    risk_level: str = "Low"
    key_issues: List[str] = field(default_factory=list)
    
    # Cleansing Recommendations
    cleansing_recommendations: List[Dict] = field(default_factory=list)

    fuzzy_keys: List[str] = field(default_factory=list)
    # New fields for enhanced analysis
    patterns: Dict = field(default_factory=dict)
    accuracy_info: Dict = field(default_factory=dict)
    validity_info: Dict = field(default_factory=dict)
    blank_count: Dict = field(default_factory=dict)
    formatting_info: Dict = field(default_factory=dict)


@dataclass
class DuplicateGroup:
    group_id: int
    indices: List[int]
    values: List[Dict]
    match_type: str
    similarity_score: Optional[float] = None
    key_columns: List[str] = field(default_factory=list)
    representative_value: Optional[str] = None


@dataclass
class DataQualityReport:
    total_rows: int
    total_columns: int
    total_cells: int
    missing_cells: int
    missing_percentage: float
    exact_duplicate_rows: int
    exact_duplicate_percentage: float
    fuzzy_duplicate_groups: int
    fuzzy_duplicate_rows: int
    columns_with_issues: List[str]
    overall_score: float
    fuzzy_match_summary: Dict = field(default_factory=dict)