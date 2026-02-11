"""Load Data Component - Enterprise file upload with chunked processing"""

import streamlit as st
import pandas as pd
import os
import time

from core.large_file_handler import ChunkedFileUploader, StreamingDataLoader
from state.session import init_session_state, recalculate_profiles, show_toast, check_profiling_complete, st


def render_load_data():
    """Enterprise file upload with progress tracking and large file support"""
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📤 Load Data (Up to 1 GB)</div>', unsafe_allow_html=True)
    
    state = st.session_state.app_state
    
    # File upload section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file (CSV, Excel, JSON, Parquet, Feather)",
            type=['csv', 'txt', 'tsv', 'xlsx', 'xls', 'json', 'jsonl', 
                  'parquet', 'pq', 'feather', 'ftr'],
            help="Supports files up to 1 GB. Large files are processed in chunks."
        )
    
    with col2:
        st.markdown("**Supported Formats**")
        st.caption("• CSV/TSV/TXT\n• Excel (XLSX/XLS)\n• JSON/JSONL\n• Parquet\n• Feather")
        
        if state.file_path and os.path.exists(state.file_path):
            file_size = os.path.getsize(state.file_path)
            st.metric("Current File", f"{file_size / 1024 / 1024:.1f} MB")
    
    # Handle upload
    if uploaded_file is not None and state.df is None:
        _handle_file_upload(uploaded_file)
    
    # Show current data status
    if state.df is not None:
        _show_data_status()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _handle_file_upload(uploaded_file):
    """Process file upload with progress tracking"""
    state = st.session_state.app_state
    
    # Progress placeholder
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize uploader
            uploader = ChunkedFileUploader()
            
            # Progress callback
            def update_progress(p):
                progress_bar.progress(int(p.percentage))
                status_text.text(f"{p.status}: {p.percentage:.1f}% - {p.message}")
            
            # Upload with progress
            status_text.text("Starting upload...")
            file_path = uploader.upload_with_progress(uploaded_file, update_progress)
            
            # Update state
            state.file_path = file_path
            state.filename = uploaded_file.name
            
            # Fast preview load
            status_text.text("Loading preview...")
            loader = StreamingDataLoader(file_path)
            
            # Show file info
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
            estimated_rows = loader.estimate_rows()
            
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("File Size", f"{file_size_mb:.1f} MB")
            with info_col2:
                st.metric("Estimated Rows", f"{estimated_rows:,}" if estimated_rows else "Unknown")
            with info_col3:
                st.metric("File Type", loader.get_file_type().upper())
            
            # Load data with optimization
            if loader.is_large_file:
                st.info("📦 Large file detected. Using streaming load...")
                
                # Load with progress
                def load_callback(processed, total, pct, msg=""):
                    progress_bar.progress(int(pct))
                    status_text.text(f"{msg} Processed: {processed:,} rows")
                
                df = loader.load_full_streaming(load_callback)
            else:
                df = loader.load_fast_preview(n_rows=None)  # Load all
            
            # Store in state
            state.original_df = df.copy()
            state.df = df
            state.processing_status = 'profiling'
            
            progress_bar.progress(100)
            status_text.text(f"✅ Upload complete! Starting data profiling... ({len(df):,} records processed)")
            
            # Trigger profiling
            recalculate_profiles()
            
            # Persist data for refresh recovery
            from state.session import _save_persisted_data
            _save_persisted_data()
            
            show_toast(f"Successfully loaded {len(df):,} rows × {len(df.columns)} columns", "success")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ Error: {str(e)}")
            show_toast(f"Upload failed: {str(e)}", "error")
            state.processing_status = 'error'


def _show_data_status():
    """Display current data status"""
    state = st.session_state.app_state
    
    st.success(f"✅ **{state.filename}** loaded successfully")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(state.df):,}")
    with col2:
        st.metric("Columns", len(state.df.columns))
    with col3:
        memory_mb = state.df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_mb:.1f} MB")
    with col4:
        status = state.processing_status
        status_emoji = "⏳" if status == 'profiling' else "✅" if status == 'ready' else "⚠️"
        st.metric("Status", f"{status_emoji} {status.title()}")
    
    # Processing status
    if state.processing_status == 'profiling':
        # Check for updates from background thread
        check_profiling_complete()
        
        msg = "🔍 Data profiling in progress... Please wait or check the Data Profiling tab."
        if state.upload_progress:
            # Calculate records processed
            total_rows = len(state.df) if state.df is not None else 0
            current_col = state.upload_progress.get('current', 0)
            total_cols = state.upload_progress.get('total', 1)
            processed_records = current_col * total_rows
            
            msg = f"🔍 Data profiling in progress... ({processed_records:,} records processed)"
            
        st.info(msg)
        
        # Show progress if available
        if state.upload_progress:
            prog = state.upload_progress
            
            # Enhanced progress display
            msg = prog['message']
            if 'elapsed' in prog:
                msg += f" | Time: {prog['elapsed']}"
            if 'eta' in prog:
                msg += f" | ETA: {prog['eta']}"
            
            # Check for rate
            if 'rate' in prog:
                 msg += f" ({prog['rate']})"
                 
            st.progress(prog['percent'] / 100, text=msg)
            
            # Auto-refresh to check status again
            time.sleep(2)
            st.rerun()
    
    # Data preview
    with st.expander("👁️ Quick Preview (First 100 rows)"):
        st.dataframe(state.df.head(100), width="stretch", height=300)
    
    # Column summary
    with st.expander("📋 Column Summary"):
        col_data = []
        for col in state.df.columns:
            dtype = str(state.df[col].dtype)
            null_count = state.df[col].isnull().sum()
            null_pct = (null_count / len(state.df)) * 100
            col_data.append({
                'Column': col,
                'Type': dtype,
                'Non-Null': f"{len(state.df) - null_count:,}",
                'Null %': f"{null_pct:.1f}%",
                'Unique': state.df[col].nunique()
            })
        
        st.dataframe(pd.DataFrame(col_data), width="stretch", hide_index=True)
    
    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reload File", width="stretch"):
            state.df = None
            state.original_df = None
            state.file_path = None
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Data", width="stretch", type="secondary"):
            from state.session import reset_application, clear_persisted_data
            clear_persisted_data()  # Clear persisted data from disk
            reset_application()
            st.rerun()