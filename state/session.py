
import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
from dataclasses import dataclass, field
import threading
import queue
import logging
import pickle
from pathlib import Path
import tempfile

# Suppress WebSocket errors in threads
logging.getLogger('tornado').setLevel(logging.CRITICAL)

# Persistence directory
PERSIST_DIR = Path(tempfile.gettempdir()) / "data_profiler_pro_cache"
PERSIST_DIR.mkdir(exist_ok=True)


@dataclass
class AppState:
    """Structured application state"""
    # Data
    df: Optional[pd.DataFrame] = None
    original_df: Optional[pd.DataFrame] = None
    filename: str = "dataset"
    file_path: Optional[str] = None
    
    # Profiling
    column_profiles: Dict = field(default_factory=dict)
    quality_report: Optional[Any] = None
    exact_duplicates: list = field(default_factory=list)
    fuzzy_duplicates: list = field(default_factory=list)
    combined_duplicates: list = field(default_factory=list)
    ai_validation_rules: Optional[pd.DataFrame] = None
    ai_validation_rules_generated: bool = False
    profiling_complete: bool = False
    
    # Operations
    fixes_applied: list = field(default_factory=list)
    operation_count: int = 0
    last_operation: Optional[str] = None
    
    # UI State
    current_tab: str = "Load Data"
    upload_progress: Optional[Dict] = None
    processing_status: str = "idle"
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'app_state': AppState(),
        'data_cache': {},
        'profiling_thread': None,
        'background_profiler': None,
        'upload_cancelled': False,
        'toast_queue': [],
        'pending_operations': [],
        'compare_state': {
            'original_sample': None,
            'modified_sample': None,
            'changes_highlighted': False
        },
        '_websocket_patched': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Ensure app_state exists
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    
    # Try to restore persisted data on first load
    _restore_persisted_data()
    
    # Backwards compatibility: ensure flat keys exist so older modules
    # that reference st.session_state.df (etc.) won't crash.
    _sync_flat_state_from_appstate()


def _get_persist_path():
    """Get the path for persisted data file"""
    return PERSIST_DIR / "session_data.pkl"


def _save_persisted_data():
    """Save current session data to disk for persistence across refreshes"""
    try:
        state = st.session_state.app_state
        
        # Only persist if we have data
        if state.df is None:
            return
        
        persist_data = {
            'df': state.df,
            'original_df': state.original_df,
            'filename': state.filename,
            'file_path': state.file_path,
            'column_profiles': state.column_profiles,
            'quality_report': state.quality_report,
            'exact_duplicates': state.exact_duplicates,
            'fuzzy_duplicates': state.fuzzy_duplicates,
            'combined_duplicates': state.combined_duplicates,
            'ai_validation_rules': state.ai_validation_rules,
            'ai_validation_rules_generated': state.ai_validation_rules_generated,
            'profiling_complete': state.profiling_complete,
            'processing_status': state.processing_status
        }
        
        with open(_get_persist_path(), 'wb') as f:
            pickle.dump(persist_data, f)
            
    except Exception as e:
        logging.debug(f"Failed to persist data: {e}")


def _restore_persisted_data():
    """Restore persisted data from disk if available"""
    try:
        persist_path = _get_persist_path()
        
        # Only restore if we don't already have data and persisted file exists
        if st.session_state.app_state.df is not None:
            return
        
        if not persist_path.exists():
            return
        
        with open(persist_path, 'rb') as f:
            persist_data = pickle.load(f)
        
        state = st.session_state.app_state
        
        # Restore all persisted fields
        state.df = persist_data.get('df')
        state.original_df = persist_data.get('original_df')
        state.filename = persist_data.get('filename', 'dataset')
        state.file_path = persist_data.get('file_path')
        state.column_profiles = persist_data.get('column_profiles', {})
        state.quality_report = persist_data.get('quality_report')
        state.exact_duplicates = persist_data.get('exact_duplicates', [])
        state.fuzzy_duplicates = persist_data.get('fuzzy_duplicates', [])
        state.combined_duplicates = persist_data.get('combined_duplicates', [])
        state.ai_validation_rules = persist_data.get('ai_validation_rules')
        state.ai_validation_rules_generated = persist_data.get('ai_validation_rules_generated', False)
        state.profiling_complete = persist_data.get('profiling_complete', False)
        state.processing_status = persist_data.get('processing_status', 'ready')
        
    except Exception as e:
        logging.debug(f"Failed to restore persisted data: {e}")
    finally:
        # Ensure flat session keys are in sync after restore attempt
        _sync_flat_state_from_appstate()


def clear_persisted_data():
    """Clear persisted data from disk"""
    try:
        persist_path = _get_persist_path()
        if persist_path.exists():
            persist_path.unlink()
    except Exception as e:
        logging.debug(f"Failed to clear persisted data: {e}")



def safe_rerun():
    """
    Safely trigger rerun, handling WebSocket disconnections
    """
    try:
        st.rerun()
    except Exception as e:
        # If WebSocket is closed, just continue
        if "WebSocket" in str(e) or "StreamClosed" in str(e):
            logging.debug("WebSocket closed during rerun, ignoring")
            pass
        else:
            raise


def show_toast(message: str, type_: str = "info"):
    """Add toast notification to queue - WebSocket safe"""
    import time
    
    # Limit toast queue size to prevent memory issues
    if len(st.session_state.toast_queue) > 10:
        st.session_state.toast_queue.pop(0)
    
    toast = {
        'message': message,
        'type': type_,
        'id': time.time(),
        'shown': False
    }
    st.session_state.toast_queue.append(toast)


def render_toasts():
    """Render pending toast notifications - with error handling"""
    to_remove = []
    
    for toast in st.session_state.toast_queue[:]:  # Copy to avoid modification during iteration
        if not toast['shown']:
            try:
                if toast['type'] == 'success':
                    st.toast(toast['message'], icon='✅')
                elif toast['type'] == 'error':
                    st.toast(toast['message'], icon='❌')
                elif toast['type'] == 'warning':
                    st.toast(toast['message'], icon='⚠️')
                else:
                    st.toast(toast['message'], icon='ℹ️')
                
                toast['shown'] = True
                to_remove.append(toast)
            except Exception as e:
                # WebSocket might be closed, mark as shown to prevent retry
                if "WebSocket" in str(e) or "StreamClosed" in str(e):
                    toast['shown'] = True
                    to_remove.append(toast)
                else:
                    raise
    
    # Clear shown toasts
    for t in to_remove:
        if t in st.session_state.toast_queue:
            st.session_state.toast_queue.remove(t)


def update_dataframe(new_df: pd.DataFrame, operation: str):
    """Update dataframe and log operation - Thread safe"""
    from datetime import datetime
    
    state = st.session_state.app_state
    
    with state._lock:
        state.df = new_df.copy()
        state.fixes_applied.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'operation': operation,
            'rows': len(new_df),
            'columns': len(new_df.columns)
        })
        state.last_operation = operation
        state.operation_count += 1
    
    # Keep legacy flat keys in sync for compatibility with older modules
    try:
        st.session_state['df'] = state.df
        st.session_state['original_df'] = state.original_df
        st.session_state['filename'] = state.filename
    except Exception:
        pass

    show_toast(f"✅ {operation}", "success")
    
    # Trigger recalculation
    recalculate_profiles()


def recalculate_profiles():
    """Recalculate all profiles with WebSocket-safe progress tracking"""
    from core.profiler import DataProfilerEngine
    
    state = st.session_state.app_state
    
    if state.df is None:
        return
    
    try:
        state.processing_status = 'profiling'
        
        # For large datasets, use background thread with minimal UI updates
        if len(state.df) > 100000:
            _start_background_profiling(state)
        else:
            # Synchronous for small datasets
            engine = DataProfilerEngine(state.df, state.filename)
            engine.profile()
            
            with state._lock:
                state.column_profiles = engine.column_profiles
                state.quality_report = engine.quality_report
                state.exact_duplicates = engine.exact_duplicates
                state.fuzzy_duplicates = engine.fuzzy_duplicates
                state.profiling_complete = True
                state.processing_status = 'ready'
            
            show_toast("Data profiling complete!", "success")
            
    except Exception as e:
        state.processing_status = 'error'
        show_toast(f"Error profiling data: {str(e)}", "error")


def _start_background_profiling(state):
    """Start background profiling with minimal WebSocket dependency"""
    
    # Create queues
    result_queue = queue.Queue()
    progress_queue = queue.Queue()
    
    def profile_callback(progress):
        try:
            # Progress is now a dictionary with rich metrics
            progress_queue.put(progress)
        except:
            pass
    
    def profile_worker():
        try:
            from core.profiler import DataProfilerEngine
            
            # Create engine
            engine = DataProfilerEngine(state.df, state.filename)
            
            # Profile with progress callback
            engine.profile(progress_callback=profile_callback)
            
            # Store result in queue instead of direct UI update
            result_queue.put(('success', engine))
            
        except Exception as e:
            result_queue.put(('error', str(e)))
    
    # Start thread
    thread = threading.Thread(target=profile_worker, daemon=True)
    thread.start()
    
    # Store reference
    st.session_state.background_profiler = {
        'thread': thread,
        'queue': result_queue,
        'progress_queue': progress_queue,
        'start_time': pd.Timestamp.now()
    }
    # ensure flat progress key exists for legacy code
    try:
        st.session_state['upload_progress'] = st.session_state.app_state.upload_progress
    except Exception:
        pass


def check_profiling_complete():
    """Check if background profiling is complete - WebSocket safe"""
    profiler_data = st.session_state.get('background_profiler')
    
    if profiler_data is None:
        return True
    
    # Drain progress queue
    try:
        while True:
            progress = profiler_data['progress_queue'].get_nowait()
            st.session_state.app_state.upload_progress = progress
    except queue.Empty:
        pass
        
    try:
        # Non-blocking check
        result = profiler_data['queue'].get_nowait()
    except queue.Empty:
        # Check if thread is still alive
        if not profiler_data['thread'].is_alive():
            # Thread died unexpectedly
            st.session_state.app_state.processing_status = 'error'
            st.session_state.background_profiler = None
            return True
        return False
    
    # Process result
    status, data = result
    state = st.session_state.app_state
    
    if status == 'success':
        engine = data
        with state._lock:
            state.column_profiles = engine.column_profiles
            state.quality_report = engine.quality_report
            state.exact_duplicates = engine.exact_duplicates
            state.fuzzy_duplicates = engine.fuzzy_duplicates
            state.profiling_complete = True
            state.processing_status = 'ready'
        show_toast("Data profiling complete!", "success")
    else:
        state.processing_status = 'error'
        show_toast(f"Profiling failed: {data}", "error")
    
    st.session_state.background_profiler = None
    return True


def reset_application():
    """Reset all application state - Safely"""
    try:
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Reinitialize
        init_session_state()
        show_toast("Application reset successfully", "success")
    except Exception as e:
        # If WebSocket is closed during reset, force clear
        st.session_state.clear()
        init_session_state()


def _sync_flat_state_from_appstate():
    """Synchronize older flat session keys from AppState for backward compatibility.

    This ensures modules that still reference `st.session_state.df` or other
    top-level keys do not raise AttributeError. It's intentionally minimal
    and non-destructive: it only sets flat keys to current AppState values.
    """
    try:
        state = st.session_state.get('app_state')
        if state is None:
            return

        # Basic dataset references
        if 'df' not in st.session_state:
            st.session_state['df'] = state.df
        else:
            # keep flat key in-sync if AppState has a value
            if state.df is not None:
                st.session_state['df'] = state.df

        # other commonly used legacy keys
        for key in ('original_df', 'filename', 'column_profiles', 'quality_report',
                    'exact_duplicates', 'fuzzy_duplicates', 'ai_validation_rules',
                    'ai_validation_rules_generated', 'profiling_complete', 
                    'processing_status', 'upload_progress', 'fixes_applied', 
                    'operation_count', 'last_operation'):
            try:
                val = getattr(state, key, None)
                if key not in st.session_state:
                    st.session_state[key] = val
                else:
                    if val is not None:
                        st.session_state[key] = val
            except Exception:
                pass
    except Exception:
        pass