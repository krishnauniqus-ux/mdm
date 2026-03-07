"""Load Data Component - Simple and intuitive file upload"""

import streamlit as st
import pandas as pd
import os
import time

from core.large_file_handler import ChunkedFileUploader, StreamingDataLoader
from state.session import init_session_state, recalculate_profiles, show_toast, check_profiling_complete


def render_load_data():
    """Simple file upload with automatic processing"""
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📤 Load Data (Up to 1 GB)</div>', unsafe_allow_html=True)
    
    state = st.session_state.app_state
    
    # File upload section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'txt', 'tsv', 'xlsx', 'xls', 'json', 'jsonl', 
                  'parquet', 'pq', 'feather', 'ftr'],
            help="Supports files up to 1 GB"
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
    
    # Show data status
    if state.df is not None:
        _show_data_status()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _handle_file_upload(uploaded_file):
    """Process file upload with automatic or manual configuration"""
    state = st.session_state.app_state
    
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
            status_text.text("Uploading file...")
            file_path = uploader.upload_with_progress(uploaded_file, update_progress)
            
            # Update state
            state.file_path = file_path
            state.filename = uploaded_file.name
            
            # Load data
            status_text.text("Loading data...")
            progress_bar.progress(50)
            
            loader = StreamingDataLoader(file_path)
            file_type = loader.get_file_type()
            
            # Clear progress and show configuration
            progress_bar.empty()
            status_text.empty()
            
            # For Excel files, show configuration
            if file_type == 'excel':
                all_sheets = loader.get_excel_sheets()
                _show_sheet_selector(loader, all_sheets)
                return
            
            # For non-Excel files, show column configuration
            else:
                _show_column_selector(loader, file_type)
                return
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ Error: {str(e)}")
            show_toast(f"Upload failed: {str(e)}", "error")
            state.processing_status = 'error'


def _show_column_selector(loader, file_type):
    """Show simple header row selection for non-Excel files"""
    state = st.session_state.app_state
    
    # Load raw preview
    try:
        raw_preview = loader.load_fast_preview(n_rows=20, header=None)
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")
        return
    
    # Show raw data
    st.markdown("**Preview Data:**")
    st.dataframe(raw_preview, use_container_width=True, height=300)
    
    # Header row selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        header_row = st.number_input(
            "Select header row (0 = first row)",
            min_value=0,
            max_value=min(50, len(raw_preview)-1),
            value=0,
            key="csv_header_row"
        )
    with col2:
        st.metric("Header", f"Row {header_row}")
    with col3:
        if st.button("🚀 Load Data", type="primary", use_container_width=True, key="csv_load_btn"):
            _load_csv_with_configuration(loader, header_row)
    
    # Show info about what will happen
    if header_row > 0:
        st.info(f"💡 Row {header_row} will be used as column headers. Rows 0-{header_row-1} will be skipped.")


def _load_csv_with_configuration(loader, header_row):
    """Load non-Excel data with selected header row"""
    state = st.session_state.app_state
    
    try:
        with st.spinner("Loading data..."):
            # Load data with selected header row (standard pandas behavior)
            # This will skip rows above header_row and use header_row as column names
            df = loader.load_full_streaming(header=header_row)
            
            # Store data
            state.df = df
            state.original_df = df.copy()
            state.header_row = header_row
            state.processing_status = 'profiling'
            
            # Trigger profiling
            recalculate_profiles()
            
            # Persist data
            from state.session import _save_persisted_data
            _save_persisted_data()
            
            show_toast(f"Loaded {len(df):,} rows × {len(df.columns)} columns", "success")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        show_toast(f"Load failed: {str(e)}", "error")


def _show_sheet_selector(loader, all_sheets):
    """Show simple sheet and header row selection for Excel files"""
    state = st.session_state.app_state
    
    # Sheet selection
    selected_sheet = st.selectbox(
        "Select Sheet",
        all_sheets,
        index=0
    )
    
    # Load raw preview
    try:
        raw_preview = loader.load_fast_preview(
            n_rows=20, 
            sheet_name=selected_sheet, 
            header=None
        )
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")
        return
    
    # Show raw data
    st.markdown("**Preview Data:**")
    st.dataframe(raw_preview, use_container_width=True, height=300)
    
    # Header row selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        header_row = st.number_input(
            "Select header row (0 = first row)",
            min_value=0,
            max_value=min(50, len(raw_preview)-1),
            value=0,
            key="excel_header_row"
        )
    with col2:
        st.metric("Header", f"Row {header_row}")
    with col3:
        if st.button("🚀 Load Data", type="primary", use_container_width=True, key="excel_load_btn"):
            _load_with_configuration(loader, selected_sheet, header_row)
    
    # Show info about what will happen
    if header_row > 0:
        st.info(f"💡 Row {header_row} will be used as column headers. Rows 0-{header_row-1} will be skipped.")


def _load_with_configuration(loader, selected_sheet, header_row):
    """Load Excel data with selected header row"""
    state = st.session_state.app_state
    
    try:
        with st.spinner("Loading data..."):
            # Load data with selected header row (standard pandas behavior)
            # This will skip rows above header_row and use header_row as column names
            df = loader.load_full_streaming(sheet_name=selected_sheet, header=header_row)
            
            # Store data
            state.df = df
            state.original_df = df.copy()
            state.selected_sheet = selected_sheet
            state.header_row = header_row
            state.processing_status = 'profiling'
            
            # Trigger profiling
            recalculate_profiles()
            
            # Persist data
            from state.session import _save_persisted_data
            _save_persisted_data()
            
            show_toast(f"Loaded {len(df):,} rows × {len(df.columns)} columns", "success")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        show_toast(f"Load failed: {str(e)}", "error")


def _show_data_status():
    """Display current data status"""
    state = st.session_state.app_state
    
    st.success(f"✅ **{state.filename}** loaded successfully")
    
    # Show profiling status
    if state.processing_status == 'profiling':
        _render_profiling_status()
    elif state.processing_status == 'ready':
        _render_ready_status()
    
    # Preview and actions
    _render_preview_and_actions()


def _render_profiling_status():
    """Show profiling progress"""
    state = st.session_state.app_state
    
    # Check for updates
    check_profiling_complete()
    
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
        st.metric("Status", "🔍 Profiling...")
    
    # Progress message
    msg = "🔍 Data profiling in progress... Please wait or check the Data Profiling tab."
    if state.upload_progress:
        total_rows = len(state.df)
        current_col = state.upload_progress.get('current', 0)
        processed_records = current_col * total_rows
        msg = f"🔍 Profiling in progress... ({processed_records:,} records processed)"
    
    st.info(msg)
    
    # Show progress bar
    if state.upload_progress:
        prog = state.upload_progress
        progress_msg = prog['message']
        if 'elapsed' in prog:
            progress_msg += f" | Time: {prog['elapsed']}"
        if 'eta' in prog:
            progress_msg += f" | ETA: {prog['eta']}"
        if 'rate' in prog:
            progress_msg += f" ({prog['rate']})"
        
        st.progress(prog['percent'] / 100, text=progress_msg)
        
        # Auto-refresh
        time.sleep(2)
        st.rerun()


def _render_ready_status():
    """Show ready status with metrics"""
    state = st.session_state.app_state
    
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
        if state.quality_report:
            score = state.quality_report.overall_score
            st.metric("Quality Score", f"{score:.0f}/100")
        else:
            st.metric("Status", "✅ Ready")
    
    st.success("✅ Data is ready! Explore other tabs for analysis.")


def _render_preview_and_actions():
    """Show preview and action buttons"""
    state = st.session_state.app_state
    
    # Data preview
    with st.expander("👁️ Data Preview (First 100 rows)", expanded=False):
        st.dataframe(state.df.head(100), use_container_width=True, height=300)
    
    # Column summary
    with st.expander("📋 Column Summary", expanded=False):
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
        
        st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)
    
    # Actions
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load Different File", use_container_width=True):
            state.df = None
            state.original_df = None
            state.file_path = None
            state.processing_status = 'idle'
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
            from state.session import reset_application, clear_persisted_data
            clear_persisted_data()
            reset_application()
            st.rerun()
