import streamlit as st
import pandas as pd
import re
from io import BytesIO

st.set_page_config(page_title="Enterprise Data Quality Tool", layout="wide")

# ===============================
# Universal Rule Engine
# ===============================

class RuleEngine:

    @staticmethod
    def apply_regex(series, pattern):
        try:
            return series.astype(str).str.match(pattern, na=False)
        except Exception as e:
            st.error(f"Regex Error: {e}")
            return pd.Series([False] * len(series))

    @staticmethod
    def apply_expression(df, column, expression):
        try:
            # Replace keyword "value" with column reference
            expr = expression.replace("value", f"`{column}`")
            return df.eval(expr)
        except Exception as e:
            st.error(f"Expression Error: {e}")
            return pd.Series([False] * len(df))


# ===============================
# UI START
# ===============================

st.title("Enterprise Data Quality Validation Tool")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully")

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.divider()

    # ===============================
    # RULE CONFIGURATION PANEL
    # ===============================

    st.subheader("Create Validation Rule")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_column = st.selectbox("Select Column", df.columns)

    with col2:
        rule_type = st.selectbox(
            "Select Rule Type",
            ["REGEX", "EXPRESSION"]
        )

    with col3:
        dimension = st.selectbox(
            "Select Dimension",
            [
                "Completeness",
                "Validity",
                "Accuracy",
                "Consistency",
                "Uniqueness",
                "Timeliness",
                "Conformity",
                "Integrity",
                "Reasonableness",
                "Precision",
                "Accessibility",
                "Relevance"
            ]
        )

    st.divider()

    if rule_type == "REGEX":
        rule_input = st.text_input(
            "Enter Regex Pattern",
            placeholder="Example: ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        )

    elif rule_type == "EXPRESSION":
        rule_input = st.text_input(
            "Enter Expression Using 'value'",
            placeholder="Example: value >= 0"
        )
        st.caption("Use keyword 'value' to refer to selected column")

    # ===============================
    # EXECUTION
    # ===============================

    if st.button("Run Validation"):

        engine = RuleEngine()

        if rule_type == "REGEX":
            result = engine.apply_regex(df[selected_column], rule_input)

        elif rule_type == "EXPRESSION":
            result = engine.apply_expression(df, selected_column, rule_input)

        df["__validation_result__"] = result

        total = len(df)
        valid = result.sum()
        invalid = total - valid
        coverage = round((valid / total) * 100, 2)

        st.divider()

        # ===============================
        # SUMMARY DASHBOARD
        # ===============================

        st.subheader("Validation Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", total)
        m2.metric("Valid Records", valid)
        m3.metric("Invalid Records", invalid)
        m4.metric("Coverage %", coverage)

        st.divider()

        # ===============================
        # INVALID RECORDS VIEW
        # ===============================

        st.subheader("Invalid Records")

        invalid_df = df[df["__validation_result__"] == False]

        if len(invalid_df) > 0:
            st.dataframe(invalid_df.drop(columns=["__validation_result__"]), use_container_width=True)

            # Download Button
            output = BytesIO()
            invalid_df.drop(columns=["__validation_result__"]).to_csv(output, index=False)
            st.download_button(
                label="Download Invalid Records",
                data=output.getvalue(),
                file_name="invalid_records.csv",
                mime="text/csv"
            )
        else:
            st.success("No Invalid Records Found")

