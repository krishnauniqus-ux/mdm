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

    col1, col2 = st.columns([2, 1])

    with col1:
        subset_cols = st.multiselect(
            "Check specific columns (empty = all):",
            state.df.columns.tolist(),
            key="exact_subset_cols"
        )

    with col2:
        st.selectbox(
            "Keep strategy:",
            ["first", "last", "none"],
            key="exact_keep_strategy"
        )

    if st.button(
        "🔍 Scan for Exact Duplicates",
        type="primary",
        width="stretch",
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
    if not duplicate_groups:
        return

    total_groups = len(duplicate_groups)
    total_rows = sum(len(g.indices) for g in duplicate_groups)

    col1, col2, col3 = st.columns(3)
    col1.metric("Duplicate Groups", total_groups)
    col2.metric("Duplicate Rows", total_rows)

    with col3:
        if st.button(
            "🗑️ Remove All Duplicates",
            width="stretch",
            key=f"remove_all_{dup_type}"
        ):
            if dup_type == "exact":
                transform_remove_exact_duplicates(keep="first")
                st.rerun()
            else:
                st.info("Bulk remove only available for exact duplicates")

    if st.button(
        "📥 Export Duplicates to Excel",
        width="stretch",
        key=f"export_{dup_type}"
    ):
        _export_duplicates_excel(duplicate_groups, dup_type)


# ------------------------------------------------------------------ #

def _export_duplicates_excel(duplicate_groups, filename_prefix):
    import io
    from datetime import datetime

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for group in duplicate_groups[:50]:
            df = pd.DataFrame(group.values)
            df.to_excel(writer, sheet_name=f"Group_{group.group_id}", index=False)

    st.download_button(
        "⬇️ Download Excel",
        data=output.getvalue(),
        file_name=f"{filename_prefix}_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{filename_prefix}"
    )


def _display_duplicate_results(duplicate_groups, dup_type):
    """Display duplicate groups with full group value preview"""

    if not duplicate_groups:
        return

    total_groups = len(duplicate_groups)
    total_rows = sum(len(g.indices) for g in duplicate_groups)

    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duplicate Groups", total_groups)
    with col2:
        st.metric("Total Duplicate Rows", total_rows)
    with col3:
        if st.button(
            "🗑️ Remove All Duplicates",
            type="secondary",
            width="stretch",
            key=f"remove_all_{dup_type}"
        ):
            if dup_type == "exact":
                transform_remove_exact_duplicates(keep="first")
                st.rerun()
            else:
                st.info(
                    "Bulk remove not available for fuzzy/combined. "
                    "Use individual group actions below."
                )

    # Export
    st.divider()
    if st.button(
        "📥 Export Duplicates to Excel",
        width="stretch",
        key=f"export_{dup_type}"
    ):
        _export_duplicates_excel(duplicate_groups, f"{dup_type}_duplicates")

    # Individual groups
    st.divider()
    st.markdown(f"**Individual Groups (showing first 20 of {total_groups})**")

    for idx, group in enumerate(duplicate_groups[:20]):
        similarity = group.similarity_score or 100
        color = (
            "#10b981" if similarity > 95
            else "#f59e0b" if similarity > 85
            else "#ef4444"
        )

        with st.expander(
            f"Group #{group.group_id} "
            f"- {len(group.indices)} rows "
            f"- {similarity:.1f}% similarity",
            expanded=False
        ):
            # Meta info
            st.markdown(f"**Match Type:** `{group.match_type}`")

            if group.key_columns:
                st.markdown(
                    f"**Key Columns:** {', '.join(group.key_columns)}"
                )

            # Similarity bar
            st.markdown(
                f"""
                <div style="height:8px;background:#e2e8f0;border-radius:4px;margin:8px 0;">
                    <div style="height:100%;width:{similarity}%;
                                background:{color};border-radius:4px;">
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 🔥 SHOW GROUP VALUES (PRESERVED)
            try:
                comparison_df = pd.DataFrame(group.values)
                st.dataframe(
                    comparison_df,
                    width="stretch",
                    key=f"df_{dup_type}_{group.group_id}"
                )

                # Actions
                action_col1, action_col2, action_col3 = st.columns(3)

                with action_col1:
                    if st.button(
                        "Keep First",
                        width="stretch",
                        key=f"{dup_type}_keep_first_{group.group_id}"
                    ):
                        transform_remove_fuzzy_group(
                            group.indices, "keep_first"
                        )
                        st.rerun()

                with action_col2:
                    if st.button(
                        "Keep Last",
                        width="stretch",
                        key=f"{dup_type}_keep_last_{group.group_id}"
                    ):
                        transform_remove_fuzzy_group(
                            group.indices, "keep_last"
                        )
                        st.rerun()

                with action_col3:
                    if st.button(
                        "Merge",
                        width="stretch",
                        key=f"{dup_type}_merge_{group.group_id}"
                    ):
                        transform_remove_fuzzy_group(
                            group.indices, "merge"
                        )
                        st.rerun()

            except Exception as e:
                st.error(f"Error displaying group: {e}")
