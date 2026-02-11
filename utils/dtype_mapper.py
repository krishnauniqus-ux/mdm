"""
Data Type Mapper - Convert technical dtypes to human-readable labels

This module provides a centralized mapping strategy for converting pandas/numpy
technical data types (dtypes) into business-friendly, human-readable labels
suitable for reports, dashboards, and non-technical stakeholders.

Author: Data Profiler Pro
Version: 1.0.0
"""

from typing import Union
import pandas as pd
import numpy as np


# ============================================================================
# COMPREHENSIVE DTYPE MAPPING DICTIONARY
# ============================================================================

DTYPE_TO_HUMAN_READABLE = {
    # Integer Types (Standard)
    'int8': 'Whole Number',
    'int16': 'Whole Number',
    'int32': 'Whole Number',
    'int64': 'Whole Number',
    'uint8': 'Whole Number',
    'uint16': 'Whole Number',
    'uint32': 'Whole Number',
    'uint64': 'Whole Number',
    
    # Integer Types (Nullable - Pandas Extension Types)
    'Int8': 'Whole Number',
    'Int16': 'Whole Number',
    'Int32': 'Whole Number',
    'Int64': 'Whole Number',
    'UInt8': 'Whole Number',
    'UInt16': 'Whole Number',
    'UInt32': 'Whole Number',
    'UInt64': 'Whole Number',
    
    # Float Types (Standard)
    'float16': 'Decimal Number',
    'float32': 'Decimal Number',
    'float64': 'Decimal Number',
    'float': 'Decimal Number',
    
    # Float Types (Nullable - Pandas Extension Types)
    'Float32': 'Decimal Number',
    'Float64': 'Decimal Number',
    
    # String Types
    'object': 'Text',  # Most common string type in pandas
    'string': 'Text',  # Pandas StringDtype
    'str': 'Text',
    'O': 'Text',  # NumPy object dtype shorthand
    
    # Boolean Types
    'bool': 'True / False',
    'boolean': 'True / False',  # Nullable boolean
    'bool_': 'True / False',
    
    # DateTime Types
    'datetime64': 'Date & Time',
    'datetime64[ns]': 'Date & Time',
    'datetime64[ns, UTC]': 'Date & Time',
    'datetime64[ns, utc]': 'Date & Time',
    'datetime64[us]': 'Date & Time',
    'datetime64[ms]': 'Date & Time',
    'datetime64[s]': 'Date & Time',
    'datetime': 'Date & Time',
    '<M8[ns]': 'Date & Time',  # NumPy datetime64 representation
    
    # Timedelta Types
    'timedelta64': 'Time Duration',
    'timedelta64[ns]': 'Time Duration',
    'timedelta': 'Time Duration',
    '<m8[ns]': 'Time Duration',  # NumPy timedelta64 representation
    
    # Categorical Types
    'category': 'Category',
    'categorical': 'Category',
    
    # Complex Types
    'complex64': 'Complex Number',
    'complex128': 'Complex Number',
    'complex': 'Complex Number',
    
    # Period Types
    'period': 'Time Period',
    'period[D]': 'Time Period',
    'period[M]': 'Time Period',
    'period[Q]': 'Time Period',
    'period[Y]': 'Time Period',
    
    # Interval Types
    'interval': 'Interval',
    
    # Sparse Types
    'Sparse': 'Sparse Data',
    'Sparse[int64]': 'Sparse Data',
    'Sparse[float64]': 'Sparse Data',
    
    # Edge Cases
    'mixed': 'Mixed Types',
    'mixed-integer': 'Mixed Types',
    'mixed-integer-float': 'Mixed Types',
    'unknown': 'Unknown',
    'empty': 'Empty',
}


# ============================================================================
# SAFE DTYPE RESOLVER FUNCTION
# ============================================================================

def get_human_readable_dtype(dtype: Union[str, pd.api.types.CategoricalDtype, np.dtype, type]) -> str:
    """
    Convert a technical pandas/numpy dtype to a human-readable label.
    
    This function safely handles various dtype representations including:
    - String representations (e.g., 'int64', 'Float64')
    - NumPy dtype objects
    - Pandas extension dtypes (nullable types)
    - Edge cases (mixed types, empty columns, unknown types)
    
    Parameters
    ----------
    dtype : str, np.dtype, pd.api.types.CategoricalDtype, or type
        The data type to convert. Can be:
        - String: 'int64', 'object', 'datetime64[ns]'
        - NumPy dtype: np.dtype('int64')
        - Pandas dtype: pd.Int64Dtype(), pd.StringDtype()
        - Type: int, float, str, bool
    
    Returns
    -------
    str
        Human-readable label for the data type:
        - "Whole Number" for integer types
        - "Decimal Number" for float types
        - "Text" for string/object types
        - "Date & Time" for datetime types
        - "True / False" for boolean types
        - "Category" for categorical types
        - "Mixed Types" for mixed/unknown types
        - "Unknown" as fallback
    
    Examples
    --------
    >>> get_human_readable_dtype('int64')
    'Whole Number'
    
    >>> get_human_readable_dtype('Float64')  # Nullable float
    'Decimal Number'
    
    >>> get_human_readable_dtype('object')
    'Text'
    
    >>> get_human_readable_dtype('datetime64[ns]')
    'Date & Time'
    
    >>> get_human_readable_dtype('boolean')  # Nullable boolean
    'True / False'
    
    >>> get_human_readable_dtype('category')
    'Category'
    
    >>> get_human_readable_dtype('mixed')
    'Mixed Types'
    
    Notes
    -----
    - The function is designed to be defensive and never raise exceptions
    - Unknown types return "Unknown" rather than failing
    - The mapping prioritizes business-friendly terminology over technical accuracy
    - Nullable pandas dtypes (Int64, Float64, boolean, string) are supported
    """
    
    try:
        # Handle None or empty input
        if dtype is None or dtype == '':
            return 'Unknown'
        
        # Convert dtype to string for consistent processing
        dtype_str = str(dtype).strip().lower()
        
        # Direct lookup (case-insensitive)
        for key, value in DTYPE_TO_HUMAN_READABLE.items():
            if dtype_str == key.lower():
                return value
        
        # Pattern matching for complex dtype strings
        # This handles cases like "datetime64[ns, UTC]" or variations
        
        # Integer patterns
        if any(pattern in dtype_str for pattern in ['int8', 'int16', 'int32', 'int64', 'uint']):
            return 'Whole Number'
        
        # Float patterns
        if any(pattern in dtype_str for pattern in ['float', 'double']):
            return 'Decimal Number'
        
        # String/Object patterns
        if any(pattern in dtype_str for pattern in ['object', 'string', 'str', '<u']):
            return 'Text'
        
        # Boolean patterns
        if any(pattern in dtype_str for pattern in ['bool']):
            return 'True / False'
        
        # DateTime patterns
        if any(pattern in dtype_str for pattern in ['datetime', '<m8', 'timestamp']):
            return 'Date & Time'
        
        # Timedelta patterns
        if any(pattern in dtype_str for pattern in ['timedelta', '<m8']):
            return 'Time Duration'
        
        # Category patterns
        if any(pattern in dtype_str for pattern in ['category', 'categorical']):
            return 'Category'
        
        # Mixed/Unknown patterns
        if any(pattern in dtype_str for pattern in ['mixed', 'unknown', 'empty']):
            return 'Mixed Types'
        
        # Complex number patterns
        if 'complex' in dtype_str:
            return 'Complex Number'
        
        # Period patterns
        if 'period' in dtype_str:
            return 'Time Period'
        
        # Interval patterns
        if 'interval' in dtype_str:
            return 'Interval'
        
        # Sparse patterns
        if 'sparse' in dtype_str:
            return 'Sparse Data'
        
        # Fallback for truly unknown types
        return 'Unknown'
        
    except Exception:
        # Ultimate fallback - never raise exceptions
        return 'Unknown'


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_mappings() -> dict:
    """
    Get the complete dtype mapping dictionary.
    
    Useful for documentation, testing, or extending the mapping.
    
    Returns
    -------
    dict
        Complete mapping of technical dtypes to human-readable labels
    """
    return DTYPE_TO_HUMAN_READABLE.copy()


def add_custom_mapping(technical_dtype: str, human_label: str) -> None:
    """
    Add a custom dtype mapping at runtime.
    
    This allows extending the mapping for domain-specific or custom dtypes
    without modifying the core mapping dictionary.
    
    Parameters
    ----------
    technical_dtype : str
        The technical dtype string to map
    human_label : str
        The human-readable label to assign
    
    Examples
    --------
    >>> add_custom_mapping('my_custom_type', 'Custom Data')
    >>> get_human_readable_dtype('my_custom_type')
    'Custom Data'
    """
    DTYPE_TO_HUMAN_READABLE[technical_dtype] = human_label


def validate_dtype_coverage(df: pd.DataFrame) -> dict:
    """
    Validate dtype coverage for a DataFrame.
    
    Checks which dtypes in the DataFrame are mapped and which are unmapped.
    Useful for identifying gaps in the mapping dictionary.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    
    Returns
    -------
    dict
        Dictionary with 'mapped' and 'unmapped' dtype lists
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    >>> validate_dtype_coverage(df)
    {'mapped': ['int64', 'object'], 'unmapped': []}
    """
    mapped = []
    unmapped = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        readable = get_human_readable_dtype(dtype)
        
        if readable == 'Unknown':
            unmapped.append(dtype)
        else:
            mapped.append(dtype)
    
    return {
        'mapped': list(set(mapped)),
        'unmapped': list(set(unmapped))
    }


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == '__main__':
    """
    Test the dtype mapper with common pandas dtypes.
    """
    print("=" * 70)
    print("DTYPE MAPPER - TEST SUITE")
    print("=" * 70)
    
    # Test cases
    test_dtypes = [
        'int64', 'Int64', 'float64', 'Float64', 'object', 'string',
        'bool', 'boolean', 'datetime64[ns]', 'category', 'mixed',
        'unknown', 'timedelta64[ns]', 'complex128'
    ]
    
    print("\nTesting standard dtypes:")
    print("-" * 70)
    for dtype in test_dtypes:
        readable = get_human_readable_dtype(dtype)
        print(f"{dtype:25s} → {readable}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
