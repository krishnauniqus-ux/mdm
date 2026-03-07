"""Export Component - Enterprise export with streaming support"""

import streamlit as st
import pandas as pd
import io
import json
from datetime import datetime

from state.session import st, show_toast
from core.large_file_handler import StreamingDataLoader


def render_export():
    """Enterprise data export with large file support"""
    
    state = st.session_state.app_state
    
    if state.df is None:
        st.info("📤 No data to export")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📥 Export Data</div>', unsafe_allow_html=True)
    
    df = state.df
    
    # Export configuration
    st.markdown("### ⚙️ Export Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_format = st.selectbox("Format:", 
                                    ["CSV", "Excel", "Parquet", "JSON", "Feather"],
                                    help="""
                                    - CSV: Universal compatibility
                                    - Excel: Best for analysis (limit 1M rows)
                                    - Parquet: Compressed, fast, preserves types
                                    - JSON: API-friendly
                                    - Feather: Fast I/O for Python
                                    """)
    
    with col2:
        # Column selection
        all_cols = df.columns.tolist()
        export_cols = st.multiselect("Columns (empty = all):", 
                                    all_cols,
                                    default=all_cols,
                                    key="export_cols")
    
    with col3:
        # Row filtering
        if len(df) > 100000:
            st.warning(f"Large dataset: {len(df):,} rows")
            sample_pct = st.slider("Sample %:", 1, 100, 100)
        else:
            sample_pct = 100
    
    # Advanced options
    with st.expander("Advanced Options"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            include_index = st.checkbox("Include Index", value=False)
            encoding = st.selectbox("Encoding:", ["utf-8", "latin-1", "utf-16"])
        
        with adv_col2:
            compression = st.selectbox("Compression:", 
                                      ["none", "gzip", "zip", "bz2"],
                                      help="For CSV/JSON only")
        
        with adv_col3:
            if export_format == "CSV":
                delimiter = st.selectbox("Delimiter:", [",", ";", "\t", "|"])
                quotechar = st.selectbox("Quote char:", ['"', "'"])
    
    # Preview export
    st.divider()
    
    # Prepare data
    if export_cols:
        export_df = df[export_cols]
    else:
        export_df = df
    
    if sample_pct < 100:
        export_df = export_df.sample(frac=sample_pct/100, random_state=42)
    
    st.markdown(f"**Export Preview:** {len(export_df):,} rows × {len(export_df.columns)} columns")

    # Safe preview: convert complex/large objects to truncated strings to avoid
    # pyarrow/json serialization MemoryError in Streamlit when rendering dataframe.
    def _safe_preview_df(df, max_cell_chars=1000, max_total_bytes=5_000_000):
        preview = df.copy()
        # Convert complex objects to JSON-safe strings and truncate long values
        total_estimated = 0
        for col in preview.columns:
            try:
                if preview[col].dtype == 'object' or preview[col].dtype == 'O':
                    def _safe_val(v):
                        try:
                            if v is None:
                                return None
                            # Convert common complex types to JSON strings
                            if isinstance(v, (dict, list, tuple, set)):
                                s = json.dumps(list(v) if isinstance(v, (set, tuple)) else v, default=str)
                            else:
                                s = str(v)
                        except Exception:
                            s = str(v)
                        # truncate
                        if len(s) > max_cell_chars:
                            return s[:max_cell_chars] + '...'
                        return s

                    preview[col] = preview[col].map(_safe_val)
                else:
                    # For non-object types, leave as-is but estimate size
                    preview[col] = preview[col]

                # estimate size
                try:
                    total_estimated += preview[col].astype(str).map(len).sum()
                except Exception:
                    total_estimated += 0

            except Exception:
                # As a fallback convert entire column to truncated strings
                preview[col] = preview[col].astype(str).str[:max_cell_chars]

        # If still too large, aggressively truncate all string cells
        if total_estimated > max_total_bytes:
            for col in preview.columns:
                try:
                    preview[col] = preview[col].astype(str).str[:200]
                except Exception:
                    preview[col] = preview[col].astype(str)

        # Reset index for preview clarity
        return preview.reset_index(drop=True)

    try:
        safe_df = _safe_preview_df(export_df.head(10))
        st.dataframe(safe_df, width=None)
    except Exception as e:
        # Final fallback: show column names only to avoid crashing
        st.error(f"Preview unavailable: {e}")
        st.write("Columns:", export_df.columns.tolist())
    
    # Generate export
    st.divider()
    
    if st.button("⬇️ Generate Export File", type="primary", width="stretch"):
        _generate_export(export_df, export_format, include_index, encoding, compression)
    
    # Batch export for very large files
    if len(df) > 500000:
        st.divider()
        st.markdown("### 📦 Batch Export (for very large files)")
        st.info("Split large dataset into multiple files")
        
        rows_per_file = st.number_input("Rows per file:", 
                                       min_value=10000, 
                                       max_value=1000000, 
                                       value=100000,
                                       step=10000)
        
        if st.button("📁 Generate Batch Export", width="stretch"):
            _generate_batch_export(df, export_format, rows_per_file, export_cols or df.columns.tolist())
    
    st.markdown('</div>', unsafe_allow_html=True)


def _generate_export(df, format_, include_index, encoding, compression):
    """Generate single export file"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"export_{timestamp}"
    
    try:
        if format_ == "CSV":
            if compression != "none":
                filename = f"{filename_base}.csv.{compression}"
                data = df.to_csv(index=include_index, encoding=encoding, compression=compression)
            else:
                filename = f"{filename_base}.csv"
                data = df.to_csv(index=include_index, encoding=encoding)
            
            mime = "text/csv"
        
        elif format_ == "Excel":
            if len(df) > 1048576:
                st.error("❌ Excel limit exceeded (1,048,576 rows). Use Parquet or CSV instead.")
                return
            
            filename = f"{filename_base}.xlsx"
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=include_index, sheet_name='Data')
            data = output.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        elif format_ == "Parquet":
            filename = f"{filename_base}.parquet"
            output = io.BytesIO()
            df.to_parquet(output, index=include_index, compression='snappy')
            data = output.getvalue()
            mime = "application/octet-stream"
        
        elif format_ == "JSON":
            filename = f"{filename_base}.json"
            if compression != "none":
                import gzip
                json_str = df.to_json(orient='records', indent=2)
                data = gzip.compress(json_str.encode())
                filename += ".gz"
            else:
                data = df.to_json(orient='records', indent=2)
            mime = "application/json"
        
        elif format_ == "Feather":
            filename = f"{filename_base}.feather"
            output = io.BytesIO()
            df.to_feather(output)
            data = output.getvalue()
            mime = "application/octet-stream"
        
        st.download_button(
            label=f"⬇️ Download {filename}",
            data=data,
            file_name=filename,
            mime=mime,
            width="stretch"
        )
        
        show_toast(f"Export ready: {filename}", "success")
        
    except Exception as e:
        st.error(f"Export failed: {e}")
        show_toast(f"Export failed: {e}", "error")


def _generate_batch_export(df, format_, rows_per_file, columns):
    """Generate multiple files for very large datasets"""
    
    import zipfile
    
    total_rows = len(df)
    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    
    st.info(f"Creating {num_files} files...")
    
    progress_bar = st.progress(0)
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i in range(num_files):
            start_idx = i * rows_per_file
            end_idx = min((i + 1) * rows_per_file, total_rows)
            
            chunk = df.iloc[start_idx:end_idx][columns]
            
            # Convert to bytes
            if format_ == "CSV":
                chunk_data = chunk.to_csv(index=False).encode('utf-8')
                filename = f"batch_{i+1:03d}_{start_idx}_{end_idx}.csv"
            elif format_ == "Parquet":
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(chunk)
                chunk_buffer = io.BytesIO()
                pq.write_table(table, chunk_buffer, compression='snappy')
                chunk_data = chunk_buffer.getvalue()
                filename = f"batch_{i+1:03d}_{start_idx}_{end_idx}.parquet"
            else:
                chunk_data = chunk.to_csv(index=False).encode('utf-8')
                filename = f"batch_{i+1:03d}_{start_idx}_{end_idx}.csv"
            
            zf.writestr(filename, chunk_data)
            progress_bar.progress((i + 1) / num_files)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.download_button(
        label=f"⬇️ Download Batch Export ({num_files} files)",
        data=zip_buffer.getvalue(),
        file_name=f"batch_export_{timestamp}.zip",
        mime="application/zip",
        width="stretch"
    )
    
    show_toast(f"Batch export created: {num_files} files", "success")