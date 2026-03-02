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
    
    # Initialize Excel config state if not exists
    if 'excel_config' not in st.session_state:
        st.session_state.excel_config = {
            'sheets': None,
            'selected_sheet': None,
            'header_row': 0,
            'preview_df': None,
            'file_path': None,
            'show_config': False,
            'last_uploaded_file': None
        }
    
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
    
    # Handle upload - check if new file uploaded
    if uploaded_file is not None and state.df is None:
        # Check if it's a new file
        current_file_hash = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.excel_config.get('last_uploaded_file') != current_file_hash:
            # New file uploaded - reset config
            st.session_state.excel_config = {
                'sheets': None,
                'selected_sheet': None,
                'header_row': 0,
                'preview_df': None,
                'file_path': None,
                'show_config': False,
                'last_uploaded_file': current_file_hash
            }
            _handle_file_upload(uploaded_file)
        elif st.session_state.excel_config.get('show_config'):
            # Show Excel configuration UI
            _render_excel_config_ui()
    
    # Show current data status
    if state.df is not None:
        _show_data_status()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _handle_file_upload(uploaded_file):
    """Process file upload with progress tracking - handle Excel sheet selection"""
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
            
            # Store in excel config
            st.session_state.excel_config['file_path'] = file_path
            
            # Check if Excel file
            loader = StreamingDataLoader(file_path)
            file_type = loader.get_file_type()
            
            # Show file info
            file_size_mb = os.path.getsize(file_path) / 1024 / 1024
            estimated_rows = loader.estimate_rows()
            
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("File Size", f"{file_size_mb:.1f} MB")
            with info_col2:
                st.metric("Estimated Rows", f"{estimated_rows:,}" if estimated_rows else "Unknown")
            with info_col3:
                st.metric("File Type", file_type.upper())
            
            if file_type == 'excel':
                # Get sheet names
                sheet_names = loader.get_excel_sheet_names()
                
                if sheet_names and len(sheet_names) > 0:
                    # Store sheets in config
                    st.session_state.excel_config['sheets'] = sheet_names
                    st.session_state.excel_config['show_config'] = True
                    
                    if len(sheet_names) == 1:
                        st.session_state.excel_config['selected_sheet'] = sheet_names[0]
                    
                    progress_bar.empty()
                    status_text.success("✅ File uploaded! Please select sheet and header row below.")
                    
                    # Show Excel config UI immediately
                    _render_excel_config_ui()
                    return
                else:
                    # No sheets found, treat as regular load
                    progress_bar.warning("⚠️ No sheets found in Excel file")
            
            # For non-Excel files or Excel without sheets - load directly
            progress_bar.empty()
            status_text.text("Loading data...")
            
            if loader.is_large_file:
                st.info("📦 Large file detected. Using streaming load...")
                
                def load_callback(processed, total, pct, msg=""):
                    progress_bar.progress(int(pct))
                    status_text.text(f"{msg} Processed: {processed:,} rows")
                
                df = loader.load_full_streaming(load_callback)
            else:
                df = loader.load_fast_preview(n_rows=None)
            
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


def _render_excel_config_ui():
    """Render Excel sheet selection and header row configuration UI"""
    state = st.session_state.app_state
    config = st.session_state.excel_config
    
    st.markdown('<div class="card" style="margin-top: 1rem; padding: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📊 Excel Configuration</div>', unsafe_allow_html=True)
    
    # Check if we have file path
    if not config.get('file_path') or not os.path.exists(config['file_path']):
        st.error("❌ File not found. Please re-upload.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    loader = StreamingDataLoader(config['file_path'])
    
    # Sheet Selection
    sheets = config.get('sheets', [])
    
    if len(sheets) > 1:
        st.markdown("**📑 Select Sheet**")
        selected_sheet = st.selectbox(
            "Choose a sheet:",
            options=sheets,
            index=sheets.index(config.get('selected_sheet', sheets[0])) if config.get('selected_sheet') in sheets else 0,
            key="excel_sheet_selector",
            help="Select the Excel sheet you want to load"
        )
        config['selected_sheet'] = selected_sheet
    elif len(sheets) == 1:
        config['selected_sheet'] = sheets[0]
        st.info(f"📑 **Sheet:** `{sheets[0]}`")
    else:
        st.warning("⚠️ No sheets found in this Excel file.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Header Row Selection
    st.markdown("---")
    st.markdown("**📋 Header Row Selection**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        header_option = st.radio(
            "Header option:",
            options=["Use first row as header (Row 0)", "Custom header row", "No header (auto-generate)"],
            index=0 if config.get('header_row', 0) == 0 else 1 if config.get('header_row', 0) >= 0 else 2,
            key="header_option_radio"
        )
    
    with col2:
        if header_option == "Use first row as header (Row 0)":
            config['header_row'] = 0
            st.success("✅ Row 0 will be used as column headers")
            
        elif header_option == "No header (auto-generate)":
            config['header_row'] = -1
            st.info("📋 Columns will be named: Column_1, Column_2, etc.")
            
        else:  # Custom header row
            header_row = st.number_input(
                "Select header row (0-indexed):",
                min_value=0,
                max_value=50,
                value=config.get('header_row', 0),
                step=1,
                key="header_row_input",
                help="Row number to use as column headers (0 = first row)"
            )
            config['header_row'] = int(header_row)
            st.success(f"✅ Row {header_row} will be used as column headers")
    
    # Preview Section
    st.markdown("---")
    st.markdown("**👁️ Preview**")
    
    try:
        preview_df = loader.preview_excel_sheet(
            config['selected_sheet'], 
            n_rows=20
        )
        
        # Show row numbers for header selection reference
        preview_with_index = preview_df.copy()
        preview_with_index.insert(0, 'Row #', range(len(preview_with_index)))
        
        st.dataframe(
            preview_with_index,
            width="stretch",
            height=250,
            use_container_width=True
        )
        
        # Highlight selected header row
        if config['header_row'] >= 0:
            st.caption(f"🎯 **Selected Header Row:** Row {config['header_row']} (shown above)")
        else:
            st.caption("🎯 **No Header:** Auto-generated column names will be used")
            
    except Exception as e:
        st.error(f"❌ Cannot preview sheet: {str(e)}")
    
    # Action Buttons
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 Reload File", use_container_width=True, type="secondary"):
            state.df = None
            state.original_df = None
            state.file_path = None
            st.session_state.excel_config = {
                'sheets': None,
                'selected_sheet': None,
                'header_row': 0,
                'preview_df': None,
                'file_path': None,
                'show_config': False,
                'last_uploaded_file': None
            }
            st.rerun()
    
    with col2:
        if st.button("✅ Load Selected Data", use_container_width=True, type="primary"):
            _load_excel_with_config()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _load_excel_with_config():
    """Load Excel data with selected sheet and header configuration"""
    state = st.session_state.app_state
    config = st.session_state.excel_config
    
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            loader = StreamingDataLoader(config['file_path'])
            
            def load_callback(progress_info):
                progress_bar.progress(progress_info.get('percent', 0))
                status_text.text(progress_info.get('message', 'Loading...'))
            
            # Load the selected sheet with header configuration
            df = loader.load_excel_sheet(
                sheet_name=config['selected_sheet'],
                header_row=config['header_row'],
                callback=load_callback
            )
            
            # Store in state
            state.original_df = df.copy()
            state.df = df
            state.processing_status = 'profiling'
            
            # Hide config UI
            config['show_config'] = False
            
            progress_bar.progress(100)
            status_text.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns from sheet '{config['selected_sheet']}'")
            
            # Trigger profiling
            recalculate_profiles()
            
            # Persist data for refresh recovery
            from state.session import _save_persisted_data
            _save_persisted_data()
            
            show_toast(f"Successfully loaded {len(df):,} rows × {len(df.columns)} columns", "success")
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ Error loading Excel data: {str(e)}")
            show_toast(f"Failed to load Excel data: {str(e)}", "error")
            state.processing_status = 'error'