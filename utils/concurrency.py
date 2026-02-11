"""
Concurrency Utilities
Shared tools for parallel processing, safe callbacks, and time estimation
"""

import time
import logging
import functools
import threading
from typing import Optional, Callable, Dict, Any, List, Union
from concurrent.futures import ThreadPoolExecutor
import os

# Configuration
MAX_WORKERS = min(8, os.cpu_count() or 4)
PROGRESS_UPDATE_PERCENT = 5

class UploadProgress:
    """Track upload/processing progress"""
    def __init__(self, total_size, uploaded_size, percentage, status, message=""):
        self.total_size = total_size
        self.uploaded_size = uploaded_size
        self.percentage = percentage
        self.status = status
        self.message = message


class WebSocketSafeCallback:
    """
    Wrapper for callbacks that handles WebSocket disconnections gracefully
    """
    
    def __init__(self, callback: Optional[Callable], throttle_seconds: float = 1.0):
        self.callback = callback
        self.throttle_seconds = throttle_seconds
        self._last_call_time = 0
        self._last_progress = 0
        self._failed = False
    
    def __call__(self, progress: Union[Dict, Any]) -> bool:
        """
        Call the wrapped callback with throttling and error handling.
        Returns True if callback was called successfully, False otherwise.
        """
        if self.callback is None or self._failed:
            return False
        
        # Throttle updates
        current_time = time.time()
        time_since_last = current_time - self._last_call_time
        
        # Convert to dict if needed (handling UploadProgress object)
        if hasattr(progress, 'status') and hasattr(progress, 'percentage'):
            progress_dict = {
                'status': progress.status,
                'percent': progress.percentage,
                'message': progress.message
            }
        else:
            progress_dict = progress
        
        percent = progress_dict.get('percent', 0)
        percent_change = abs(percent - self._last_progress)
        
        # Only update if enough time passed or significant progress made
        # Always update if complete or error
        status = progress_dict.get('status')
        if status not in ['complete', 'error']:
            if time_since_last < self.throttle_seconds and percent_change < PROGRESS_UPDATE_PERCENT:
                return True  # Skip but don't fail
        
        try:
            self.callback(progress_dict if isinstance(progress, dict) else progress)
            self._last_call_time = current_time
            self._last_progress = percent
            return True
            
        except Exception as e:
            # WebSocket likely closed or other error
            error_msg = str(e).lower()
            if any(x in error_msg for x in ['websocket', 'streamclosed', 'broken pipe', 'connection']):
                self._failed = True
                logging.debug(f"WebSocket callback failed, suppressing further updates: {e}")
            else:
                # Re-raise non-WebSocket errors
                pass
            return False
    
    def is_active(self) -> bool:
        """Check if callback is still active (hasn't failed)"""
        return not self._failed


class ParallelProcessor:
    """
    Parallel processing utility for CPU-intensive operations
    with WebSocket-safe progress reporting
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or MAX_WORKERS
        self._executor = None
    
    def __enter__(self):
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
    
    def map(self, func: Callable, items: List[Any], 
            progress_callback: Optional[Callable] = None,
            start_pct: int = 0, end_pct: int = 100) -> List[Any]:
        """
        Apply function to all items in parallel with progress tracking
        """
        safe_callback = WebSocketSafeCallback(progress_callback, throttle_seconds=0.5) if progress_callback else None
        total = len(items)
        if total == 0:
            return []

        completed = 0
        results = []
        
        # Submit all tasks
        # Provide index to sort later
        futures = {
            self._executor.submit(func, item): i 
            for i, item in enumerate(items)
        }
        
        # Collect results as they complete
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append((futures[future], result))
                completed += 1
                
                # Update progress
                if safe_callback:
                    # Map completed count to percentage range [start_pct, end_pct]
                    current_pct = start_pct + int((completed / total) * (end_pct - start_pct))
                    safe_callback({
                        'status': 'processing',
                        'percent': current_pct,
                        'message': f'Processed {completed}/{total} items'
                    })
                
            except Exception as e:
                # Log error but continue
                logging.error(f"Error in parallel processing: {e}")
                results.append((futures[future], None)) # Or handle differently
        
        # Sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class TimeEstimator:
    """
    Estimates remaining time for long-running processes
    """
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.start_time = time.time()
        self.last_update = 0
    
    def get_metrics(self, current_items: int) -> Dict[str, str]:
        """
        Calculate elapsed time, rate, and ETA
        Returns formatted strings
        """
        now = time.time()
        elapsed_seconds = now - self.start_time
        
        if current_items <= 0:
            return {
                'elapsed': self._format_time(elapsed_seconds),
                'eta': 'calculating...',
                'rate': '0 items/s'
            }
        
        # Calculate rate
        rate = current_items / max(elapsed_seconds, 0.001)
        
        # Calculate ETA
        remaining_items = max(0, self.total_items - current_items)
        eta_seconds = remaining_items / rate if rate > 0 else 0
        
        return {
            'elapsed': self._format_time(elapsed_seconds),
            'eta': self._format_time(eta_seconds),
            'rate': f"{rate:.1f} items/s"
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into readable string (e.g. '1m 30s')"""
        if seconds < 60:
            return f"{int(seconds)}s"
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        
        if minutes < 60:
            return f"{minutes}m {secs}s"
        
        hours = int(minutes // 60)
        minutes = minutes % 60
        return f"{hours}h {minutes}m"
