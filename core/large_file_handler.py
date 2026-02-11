"""
Enterprise Large File Handler
Supports files up to 1GB with chunked uploads, streaming, and background processing
"""

import os
import io
import hashlib
import tempfile
from typing import Iterator, Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
import logging

import pandas as pd
import numpy as np
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Suppress WebSocket-related logging
logging.getLogger('tornado').setLevel(logging.CRITICAL)
logging.getLogger('streamlit').setLevel(logging.WARNING)

# Configuration
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
MEMORY_THRESHOLD = 500 * 1024 * 1024  # 500MB - switch to disk-based processing
PROGRESS_UPDATE_INTERVAL = 5  # seconds between progress updates
from utils.concurrency import UploadProgress, WebSocketSafeCallback, ParallelProcessor, MAX_WORKERS, PROGRESS_UPDATE_PERCENT


class ChunkedFileUploader:
    """Handle large file uploads in chunks with progress tracking"""
    
    def __init__(self):
        self.progress = UploadProgress(0, 0, 0.0, 'idle')
        self._cancel_event = threading.Event()
        self._temp_dir = tempfile.mkdtemp()
        self._safe_callback: Optional[WebSocketSafeCallback] = None
        
    def upload_with_progress(self, uploaded_file: UploadedFile, 
                           progress_callback: Optional[Callable] = None) -> str:
        """
        Save uploaded file with progress tracking
        Returns path to saved file
        """
        # Wrap callback for WebSocket safety
        self._safe_callback = WebSocketSafeCallback(progress_callback, throttle_seconds=0.5)
        
        self.progress = UploadProgress(
            total_size=len(uploaded_file.getvalue()),
            uploaded_size=0,
            percentage=0.0,
            status='uploading'
        )
        
        # Generate safe filename
        file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
        safe_name = f"{file_hash}_{uploaded_file.name}"
        file_path = os.path.join(self._temp_dir, safe_name)
        
        # Stream to disk in chunks
        try:
            with open(file_path, 'wb') as f:
                uploaded_file.seek(0)
                bytes_since_update = 0
                
                while True:
                    if self._cancel_event.is_set():
                        raise InterruptedError("Upload cancelled")
                    
                    chunk = uploaded_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    self.progress.uploaded_size += len(chunk)
                    bytes_since_update += len(chunk)
                    self.progress.percentage = (self.progress.uploaded_size / self.progress.total_size) * 100
                    
                    # Update progress (throttled by safe callback)
                    self._safe_callback(self.progress)
                    
                    # Small yield to prevent blocking
                    if bytes_since_update > 50 * 1024 * 1024:  # Every 50MB
                        time.sleep(0.001)
                        bytes_since_update = 0
            
            self.progress.status = 'complete'
            self.progress.message = f"File saved: {safe_name}"
            self._safe_callback(self.progress)  # Final update
            return file_path
            
        except Exception as e:
            self.progress.status = 'error'
            self.progress.message = str(e)
            if self._safe_callback:
                self._safe_callback(self.progress)  # Error update
            raise
    
    def cancel(self):
        """Cancel ongoing upload"""
        self._cancel_event.set()


class StreamingDataLoader:
    """Load large datasets with streaming and memory optimization"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.is_large_file = self.file_size > MEMORY_THRESHOLD
        
    def get_file_type(self) -> str:
        """Detect file type from extension"""
        ext = os.path.splitext(self.file_path)[1].lower()
        type_map = {
            '.csv': 'csv', '.txt': 'csv', '.tsv': 'tsv',
            '.xlsx': 'excel', '.xls': 'excel',
            '.json': 'json', '.jsonl': 'jsonl',
            '.parquet': 'parquet', '.pq': 'parquet',
            '.feather': 'feather', '.ftr': 'feather'
        }
        return type_map.get(ext, 'unknown')
    
    def estimate_rows(self) -> Optional[int]:
        """Estimate row count without loading full file"""
        file_type = self.get_file_type()
        
        if file_type == 'csv':
            # Sample first 1MB to estimate
            sample_size = min(1024 * 1024, self.file_size)
            with open(self.file_path, 'rb') as f:
                sample = f.read(sample_size)
            
            # Count newlines in sample
            newline_count = sample.count(b'\n')
            if newline_count > 0:
                bytes_per_row = len(sample) / newline_count
                estimated_rows = int(self.file_size / bytes_per_row)
                return estimated_rows
        
        return None
    
    def load_fast_preview(self, n_rows: int = 1000) -> pd.DataFrame:
        """Load quick preview for immediate display"""
        file_type = self.get_file_type()
        
        if file_type == 'csv':
            # Use low_memory=False for better type inference on preview
            return pd.read_csv(self.file_path, nrows=n_rows, low_memory=False)
        elif file_type == 'tsv':
            return pd.read_csv(self.file_path, sep='\t', nrows=n_rows, low_memory=False)
        elif file_type == 'excel':
            return pd.read_excel(self.file_path, nrows=n_rows)
        elif file_type == 'json':
            return pd.read_json(self.file_path, nrows=n_rows)
        elif file_type == 'parquet':
            # Parquet is efficient, can read metadata without full load
            return pd.read_parquet(self.file_path).head(n_rows)
        
        raise ValueError(f"Unsupported file type: {file_type}")
    
    def load_full_streaming(self, callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Load full dataset with streaming for large files
        Uses chunked reading and parallel processing
        """
        # Wrap callback for WebSocket safety
        safe_callback = WebSocketSafeCallback(callback, throttle_seconds=1.0)
        
        file_type = self.get_file_type()
        
        if not self.is_large_file:
            # Small file - load directly
            return self._load_complete(file_type)
        
        # Large file - use chunked processing
        return self._load_large_file(file_type, safe_callback)
    
    def _load_complete(self, file_type: str) -> pd.DataFrame:
        """Load entire file at once"""
        if file_type == 'csv':
            return pd.read_csv(self.file_path, low_memory=False)
        elif file_type == 'tsv':
            return pd.read_csv(self.file_path, sep='\t', low_memory=False)
        elif file_type == 'excel':
            return pd.read_excel(self.file_path)
        elif file_type == 'json':
            return pd.read_json(self.file_path)
        elif file_type == 'parquet':
            return pd.read_parquet(self.file_path)
        elif file_type == 'feather':
            return pd.read_feather(self.file_path)
        
        raise ValueError(f"Unsupported file type: {file_type}")
    
    def _load_large_file(self, file_type: str, callback: WebSocketSafeCallback) -> pd.DataFrame:
        """Load large file in chunks with parallel processing"""
        if file_type not in ['csv', 'tsv']:
            # For non-CSV, fall back to complete load with memory optimization
            return self._load_complete(file_type)
        
        sep = '\t' if file_type == 'tsv' else ','
        
        # Get total rows estimate
        total_rows = self.estimate_rows() or 0
        
        # Read in chunks
        chunk_iter = pd.read_csv(
            self.file_path, 
            sep=sep, 
            chunksize=100000,
            low_memory=False
        )
        
        chunks = []
        processed_rows = 0
        start_time = time.time()
        
        for i, chunk in enumerate(chunk_iter):
            chunks.append(chunk)
            processed_rows += len(chunk)
            
            # Calculate progress
            if total_rows > 0:
                progress_pct = min(95, (processed_rows / total_rows) * 95)  # Reserve 5% for combining
            else:
                progress_pct = min(95, (i * 100000 / (self.file_size / 100)) * 95)
            
            # Update callback (throttled internally)
            callback({
                'status': 'loading',
                'percent': progress_pct,
                'message': f'Loaded {processed_rows:,} rows...'
            })
            
            # Periodic garbage collection and yield
            if i % 10 == 0:
                import gc
                gc.collect()
        
        # Combine chunks efficiently
        callback({
            'status': 'processing',
            'percent': 95,
            'message': 'Combining chunks...'
        })
        
        result = pd.concat(chunks, ignore_index=True)
        
        callback({
            'status': 'complete',
            'percent': 100,
            'message': f'Loaded {len(result):,} rows successfully'
        })
        
        return result


class BackgroundProfiler:
    """Run data profiling in background with progress tracking"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.result_queue = queue.Queue()
        self.progress = {'status': 'idle', 'percent': 0, 'message': ''}
        self._cancel_event = threading.Event()
        self._cancelled = False
        self._last_update_time = 0
        self._last_percent = 0
        
    def run_profiling(self, callback: Optional[Callable] = None):
        """Run profiling in background thread with WebSocket-safe callbacks"""
        
        # Wrap callback for WebSocket safety and throttling
        safe_callback = WebSocketSafeCallback(callback, throttle_seconds=2.0)
        
        def profile_worker():
            try:
                from core.profiler import DataProfilerEngine
                
                self.progress = {'status': 'running', 'percent': 5, 'message': 'Initializing profiler...'}
                safe_callback(self.progress)
                
                # Initialize profiler
                engine = DataProfilerEngine(self.df, "dataset")
                
                # Profile in stages with throttled progress updates
                # We defer parallel logic to the DataProfilerEngine itself
                # But we can provide a progress callback wrapper that bridges it to our safe_callback
                
                def engine_progress_callback(progress_info):
                    # Progress info from engine is a dict
                    self.progress = progress_info
                    safe_callback(progress_info)
                
                # Run profiling (engine handles parallelism internally now)
                engine.profile(progress_callback=engine_progress_callback)

                # Profiling includes quality report generation which finds duplicates
                # No need to call find_exact_duplicates again here
                
                # Final update
                self.progress = {
                    'status': 'complete', 
                    'percent': 100, 
                    'message': f'Profiling complete - {len(engine.column_profiles)} columns analyzed'
                }
                safe_callback(self.progress)
                
                # Put result in queue
                self.result_queue.put(('success', engine))
                
            except InterruptedError:
                self.progress = {'status': 'cancelled', 'percent': 0, 'message': 'Profiling cancelled'}
                self.result_queue.put(('cancelled', None))
                
            except Exception as e:
                error_msg = str(e)
                self.progress = {'status': 'error', 'percent': 0, 'message': error_msg}
                try:
                    safe_callback(self.progress)
                except Exception:
                    pass
                self.result_queue.put(('error', error_msg))
        
        # Start background thread
        self._cancelled = False
        self._last_update_time = time.time()
        thread = threading.Thread(target=profile_worker, daemon=True)
        thread.start()
        return thread
    
    def cancel(self):
        """Cancel profiling"""
        self._cancelled = True
        self._cancel_event.set()
    
    def get_result(self, timeout: float = 0.1) -> Optional[tuple]:
        """Non-blocking result check"""
        try:
            return self.result_queue.get(block=False)
        except queue.Empty:
            return None
    
    def get_progress(self) -> Dict:
        """Get current progress"""
        return self.progress.copy()


class DataCache:
    """LRU cache for data operations"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cache = {}
                    cls._instance._access_times = {}
                    cls._instance._max_size = 10
        return cls._instance
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached item"""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
        return None
    
    def set(self, key: str, value: Any, max_size: Optional[int] = None):
        """Cache item with LRU eviction"""
        max_size = max_size or self._max_size
        
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= max_size:
                if not self._access_times:
                    break
                oldest = min(self._access_times, key=self._access_times.get)
                del self._cache[oldest]
                del self._access_times[oldest]
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    def clear(self):
        """Clear all cached data"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'keys': list(self._cache.keys())
            }



def estimate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate memory usage of dataframe in various formats
    """
    # In-memory size
    mem_bytes = df.memory_usage(deep=True).sum()
    
    # Estimate serialized sizes
    csv_buffer = io.StringIO()
    df.head(1000).to_csv(csv_buffer, index=False)
    csv_size_per_row = len(csv_buffer.getvalue().encode()) / min(1000, len(df))
    estimated_csv_size = csv_size_per_row * len(df)
    
    return {
        'memory_mb': mem_bytes / 1024 / 1024,
        'estimated_csv_mb': estimated_csv_size / 1024 / 1024,
        'estimated_parquet_mb': mem_bytes / 1024 / 1024 / 3,  # Parquet is ~3x compressed
        'row_count': len(df),
        'column_count': len(df.columns)
    }


def optimize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize dataframe memory usage by downcasting types
    """
    optimized = df.copy()
    
    for col in optimized.columns:
        col_type = optimized[col].dtype
        
        if col_type != object:
            c_min = optimized[col].min()
            c_max = optimized[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized[col] = optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized[col] = optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized[col] = optimized[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized[col] = optimized[col].astype(np.float32)
        
        else:
            # Convert objects to category if beneficial
            num_unique_values = len(optimized[col].unique())
            num_total_values = len(optimized[col])
            if num_unique_values / num_total_values < 0.5:
                optimized[col] = optimized[col].astype('category')
    
    return optimized
