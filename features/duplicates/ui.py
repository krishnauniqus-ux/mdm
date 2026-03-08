"""Find Duplicates Component - Standalone duplicate detection tab"""

import streamlit as st
import pandas as pd

from state.session import show_toast
from core.profiler import DataProfilerEngine
from core.transformations import (
    transform_remove_exact_duplicates,
    transform_remove_fuzzy_group
)


def render_find_duplicates():
    """Find Duplicates - Exact, Fuzzy, and Combined matching"""

    state = st.session_state.app_state

    if state.df is None:
        st.info("📤 Please load data first in the 'Load Data' tab")
        return

    st.markdown("## 🔍 Find Duplicates")

    dup_tabs = st.tabs([
        "📋 Exact Match",
        "🔎 Fuzzy Match",
        "🎯 Combined Match"
    ])

    with dup_tabs[0]:
        _render_exact_duplicates()

    with dup_tabs[1]:
        _render_fuzzy_duplicates()

    with dup_tabs[2]:
        _render_combined_duplicates()


# ------------------------------------------------------------------ #

def _render_exact_duplicates():
    state = st.session_state.app_state

    st.subheader("📋 Exact Duplicate Detection")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        subset_cols = st.multiselect(
            "Check specific columns (empty = all):",
            state.df.columns.tolist(),
            key="exact_subset_cols"
        )

    with col2:
        keep_strategy = st.selectbox(
            "Keep strategy:",
            ["first", "last", "none"],
            key="exact_keep_strategy",
            help="Strategy for bulk removal: keep first occurrence, last occurrence, or remove all duplicates"
        )

    with col3:
        st.markdown("&nbsp;")  # Spacer
        st.markdown("&nbsp;")

    if st.button(
        "🔍 Scan for Exact Duplicates",
        type="primary",
        use_container_width=True,
        key="scan_exact"
    ):
        with st.spinner("Scanning..."):
            engine = DataProfilerEngine(state.df, state.filename)
            state.exact_duplicates = engine.find_exact_duplicates(
                subset=subset_cols if subset_cols else None
            )
            show_toast(
                f"Found {len(state.exact_duplicates)} exact duplicate groups",
                "success"
            )

    if getattr(state, "exact_duplicates", None):
        _display_duplicate_results(state.exact_duplicates, "exact")
    else:
        st.success("✅ No exact duplicates found")


# ------------------------------------------------------------------ #

def _render_fuzzy_duplicates():
    state = st.session_state.app_state

    st.subheader("🔎 Fuzzy Duplicate Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        fuzzy_cols = st.multiselect(
            "Columns to scan:",
            state.df.select_dtypes(include=["object"]).columns.tolist(),
            key="fuzzy_cols"
        )

    with col2:
        threshold = st.slider(
            "Similarity Threshold:",
            50, 100, 85,
            key="fuzzy_threshold"
        )

    with col3:
        algorithm = st.selectbox(
            "Algorithm:",
            ["rapidfuzz", "jaro_winkler", "metaphone", "combined"],
            key="fuzzy_algo"
        )

    if st.button(
        "🔍 Scan for Fuzzy Duplicates",
        type="primary",
        width="stretch",
        key="scan_fuzzy"
    ):
        if not fuzzy_cols:
            st.warning("Please select at least one column")
        else:
            with st.spinner("Scanning..."):
                engine = DataProfilerEngine(state.df, state.filename)
                state.fuzzy_duplicates = engine.find_fuzzy_duplicates(
                    fuzzy_cols, threshold, algorithm
                )
                show_toast(
                    f"Found {len(state.fuzzy_duplicates)} fuzzy duplicate groups",
                    "success"
                )

    if getattr(state, "fuzzy_duplicates", None):
        _display_duplicate_results(state.fuzzy_duplicates, "fuzzy")
    else:
        st.info("No fuzzy duplicates found")


# ------------------------------------------------------------------ #

def _render_combined_duplicates():
    state = st.session_state.app_state

    st.subheader("🎯 Combined Duplicate Detection")

    col1, col2 = st.columns(2)

    with col1:
        exact_cols = st.multiselect(
            "Exact match columns:",
            state.df.columns.tolist(),
            key="combined_exact_cols"
        )

    with col2:
        fuzzy_cols = st.multiselect(
            "Fuzzy match columns:",
            state.df.select_dtypes(include=["object"]).columns.tolist(),
            key="combined_fuzzy_cols"
        )

    threshold = st.slider(
        "Fuzzy Threshold:",
        50, 100, 85,
        key="combined_threshold"
    )

    algorithm = st.selectbox(
        "Algorithm:",
        ["rapidfuzz", "jaro_winkler", "metaphone", "combined"],
        key="combined_algo"
    )

    if st.button(
        "🔍 Scan for Combined Duplicates",
        type="primary",
        width="stretch",
        key="scan_combined"
    ):
        engine = DataProfilerEngine(state.df, state.filename)
        state.combined_duplicates = engine.find_combined_duplicates(
            exact_cols, fuzzy_cols, threshold, algorithm
        )
        show_toast(
            f"Found {len(state.combined_duplicates)} combined duplicate groups",
            "success"
        )

    if getattr(state, "combined_duplicates", None):
        _display_duplicate_results(state.combined_duplicates, "combined")


# ------------------------------------------------------------------ #

def _display_duplicate_results(duplicate_groups, dup_type):
    """Display duplicate groups in tabular format with sheet-wise organization"""

    if not duplicate_groups:
        return

    total_groups = len(duplicate_groups)
    total_rows = sum(len(g.indices) for g in duplicate_groups)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duplicate Groups", total_groups)
    with col2:
        st.metric("Total Duplicate Rows", total_rows)
    with col3:
        if st.button(
            "🗑️ Remove All Duplicates",
            type="secondary",
            use_container_width=True,
            key=f"remove_all_{dup_type}"
        ):
            if dup_type == "exact":
                # Get the keep strategy from session state
                keep_strategy = st.session_state.get("exact_keep_strategy", "first")
                transform_remove_exact_duplicates(keep=keep_strategy)
                st.rerun()
            else:
                st.info(
                    "Bulk remove not available for fuzzy/combined. "
                    "Use sheet-wise actions below."
                )
    with col4:
        if st.button(
            "📥 Export All to Excel",
            use_container_width=True,
            key=f"export_all_{dup_type}"
        ):
            _export_duplicates_excel(duplicate_groups, f"{dup_type}_duplicates")

    st.divider()

    # Create summary table for all groups
    st.markdown("### 📊 Duplicate Groups Summary")
    
    summary_data = []
    for group in duplicate_groups:
        similarity = group.similarity_score or 100
        summary_data.append({
            'Select': False,
            'Group ID': f"Group #{group.group_id}",
            'Rows': len(group.indices),
            'Similarity': f"{similarity:.1f}%",
            'Match Type': group.match_type,
            'Key Columns': ', '.join(group.key_columns) if group.key_columns else 'All',
            'Preview': str(group.representative_value)[:50] + '...' if len(str(group.representative_value)) > 50 else str(group.representative_value)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Use data_editor for selection
    edited_summary = st.data_editor(
        summary_df,
        hide_index=True,
        use_container_width=True,
        key=f"summary_editor_{dup_type}",
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select groups to process",
                default=False,
            ),
            "Group ID": st.column_config.TextColumn("Group ID", width="small"),
            "Rows": st.column_config.NumberColumn("Rows", width="small"),
            "Similarity": st.column_config.TextColumn("Similarity", width="small"),
            "Match Type": st.column_config.TextColumn("Match Type", width="small"),
            "Key Columns": st.column_config.TextColumn("Key Columns", width="medium"),
            "Preview": st.column_config.TextColumn("Preview", width="large"),
        },
        disabled=[col for col in summary_df.columns if col != 'Select']
    )
    
    selected_groups = edited_summary[edited_summary['Select'] == True]
    
    if len(selected_groups) > 0:
        st.info(f"✓ {len(selected_groups)} group(s) selected")
        
        # Bulk actions for selected groups
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        with action_col1:
            if st.button("Keep First (Selected)", use_container_width=True, key=f"bulk_first_{dup_type}"):
                _process_selected_groups(duplicate_groups, selected_groups, "keep_first")
                st.rerun()
        
        with action_col2:
            if st.button("Keep Last (Selected)", use_container_width=True, key=f"bulk_last_{dup_type}"):
                _process_selected_groups(duplicate_groups, selected_groups, "keep_last")
                st.rerun()
        
        with action_col3:
            if st.button("Merge (Selected)", use_container_width=True, key=f"bulk_merge_{dup_type}"):
                _process_selected_groups(duplicate_groups, selected_groups, "merge")
                st.rerun()
        
        with action_col4:
            if st.button("Export Selected", use_container_width=True, key=f"export_selected_{dup_type}"):
                selected_group_ids = [int(gid.split('#')[1]) for gid in selected_groups['Group ID']]
                selected_group_objs = [g for g in duplicate_groups if g.group_id in selected_group_ids]
                _export_duplicates_excel(selected_group_objs, f"{dup_type}_selected")

    st.divider()

    # Sheet-wise detailed view
    st.markdown("### 📋 Detailed View by Group")
    
    # Create tabs for groups (show first 10 in tabs, rest in expanders)
    if total_groups <= 10:
        tab_labels = [f"Group #{g.group_id}" for g in duplicate_groups[:10]]
        tabs = st.tabs(tab_labels)
        
        for idx, (tab, group) in enumerate(zip(tabs, duplicate_groups[:10])):
            with tab:
                _render_group_detail(group, dup_type)
    else:
        # Show first 5 in tabs, rest in a scrollable section
        tab_labels = [f"Group #{g.group_id}" for g in duplicate_groups[:5]]
        tab_labels.append("More Groups...")
        tabs = st.tabs(tab_labels)
        
        for idx, (tab, group) in enumerate(zip(tabs[:-1], duplicate_groups[:5])):
            with tab:
                _render_group_detail(group, dup_type)
        
        with tabs[-1]:
            st.markdown("**Additional Groups:**")
            for group in duplicate_groups[5:20]:  # Show up to 20 total
                with st.expander(f"Group #{group.group_id} - {len(group.indices)} rows - {group.similarity_score:.1f}% similarity"):
                    _render_group_detail(group, dup_type)


def _render_group_detail(group, dup_type):
    """Render detailed view of a single duplicate group"""
    try:
        # Group metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows in Group", len(group.indices))
        with col2:
            st.metric("Similarity", f"{group.similarity_score or 100:.1f}%")
        with col3:
            st.metric("Match Type", group.match_type)
        
        if group.key_columns:
            st.caption(f"**Key Columns:** {', '.join(group.key_columns)}")
        
        # Create dataframe with checkbox column
        comparison_df = pd.DataFrame(group.values)
        comparison_df.insert(0, '✓ Keep', False)
        
        # Use data_editor for row selection
        edited_df = st.data_editor(
            comparison_df,
            hide_index=False,
            use_container_width=True,
            key=f"editor_{dup_type}_{group.group_id}",
            column_config={
                "✓ Keep": st.column_config.CheckboxColumn(
                    "✓ Keep",
                    help="Check to keep this row",
                    default=False,
                )
            },
            disabled=[col for col in comparison_df.columns if col != '✓ Keep']
        )
        
        # Get selected rows
        selected_rows = edited_df[edited_df['✓ Keep'] == True].index.tolist()
        
        if selected_rows:
            st.success(f"✓ {len(selected_rows)} row(s) selected to keep")

        # Action buttons
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)

        with action_col1:
            if st.button(
                "Keep First",
                use_container_width=True,
                key=f"{dup_type}_keep_first_{group.group_id}"
            ):
                transform_remove_fuzzy_group(group.indices, "keep_first")
                st.rerun()

        with action_col2:
            if st.button(
                "Keep Last",
                use_container_width=True,
                key=f"{dup_type}_keep_last_{group.group_id}"
            ):
                transform_remove_fuzzy_group(group.indices, "keep_last")
                st.rerun()

        with action_col3:
            if st.button(
                "Keep Selected",
                type="primary",
                use_container_width=True,
                key=f"{dup_type}_keep_selected_{group.group_id}",
                disabled=len(selected_rows) == 0
            ):
                if len(selected_rows) == 0:
                    st.warning("Please select at least one row to keep")
                elif len(selected_rows) == 1:
                    transform_remove_fuzzy_group(
                        group.indices, "keep_selected", selected_index=selected_rows[0]
                    )
                    st.rerun()
                else:
                    transform_remove_fuzzy_group(
                        group.indices, "keep_multiple", selected_indices=selected_rows
                    )
                    st.rerun()

        with action_col4:
            if st.button(
                "Merge",
                use_container_width=True,
                key=f"{dup_type}_merge_{group.group_id}"
            ):
                transform_remove_fuzzy_group(group.indices, "merge")
                st.rerun()

    except Exception as e:
        st.error(f"Error displaying group: {e}")


def _process_selected_groups(all_groups, selected_summary, strategy):
    """Process multiple selected groups with the same strategy"""
    try:
        # Extract group IDs from selected summary
        selected_group_ids = [int(gid.split('#')[1]) for gid in selected_summary['Group ID']]
        
        # Find corresponding group objects
        groups_to_process = [g for g in all_groups if g.group_id in selected_group_ids]
        
        total_removed = 0
        for group in groups_to_process:
            transform_remove_fuzzy_group(group.indices, strategy)
            if strategy in ["keep_first", "keep_last", "merge"]:
                total_removed += len(group.indices) - 1
        
        show_toast(f"Processed {len(groups_to_process)} groups, removed {total_removed} rows", "success")
        
    except Exception as e:
        st.error(f"Error processing groups: {e}")


def _export_duplicates_excel(duplicate_groups, filename_prefix):
    import io
    from datetime import datetime

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary sheet
        summary_data = []
        for group in duplicate_groups:
            summary_data.append({
                'Group ID': group.group_id,
                'Rows': len(group.indices),
                'Similarity': f"{group.similarity_score or 100:.1f}%",
                'Match Type': group.match_type,
                'Key Columns': ', '.join(group.key_columns) if group.key_columns else 'All'
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Individual group sheets (limit to 50 to avoid Excel sheet limit)
        for group in duplicate_groups[:50]:
            df = pd.DataFrame(group.values)
            sheet_name = f"Group_{group.group_id}"[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    st.download_button(
        "⬇️ Download Excel",
        data=output.getvalue(),
        file_name=f"{filename_prefix}_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{filename_prefix}_{datetime.now():%H%M%S}"
    )
