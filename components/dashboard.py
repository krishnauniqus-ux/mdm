# render_dashboard
"""Dashboard component"""

import streamlit as st
import pandas as pd


def render_dashboard():
    """Fixed Dashboard with proper error handling"""
    try:
        # Check if data is loaded
        if st.session_state.df is None:
            st.markdown('<div class="info-box">📤 Please upload a data file to view the dashboard</div>', unsafe_allow_html=True)
            return
        
        # Check if quality report exists
        if st.session_state.quality_report is None:
            with st.spinner("🔄 Calculating metrics..."):
                from state.session import recalculate_profiles
                recalculate_profiles()
        
        qr = st.session_state.quality_report
        
        if qr is None:
            st.markdown('<div class="error-box">⚠️ Unable to generate quality report. Please try re-uploading the file.</div>', unsafe_allow_html=True)
            return
        
        # Score Section
        score = qr.overall_score
        score_color = "#10b981" if score >= 90 else "#3b82f6" if score >= 75 else "#f59e0b" if score >= 60 else "#ef4444"
        status_text = "Excellent" if score >= 90 else "Good" if score >= 75 else "Fair" if score >= 60 else "Critical"
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="score-container">
                <div class="score-value" style="color: white;">{score:.0f}</div>
                <div class="score-label">Quality Score - {status_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Metrics
        st.markdown("### 📊 Key Metrics")
        cols = st.columns(4)
        
        metrics = [
            (f"{qr.total_rows:,}", "Total Rows", "#3b82f6"),
            (f"{qr.total_columns}", "Columns", "#8b5cf6"),
            (f"{qr.missing_percentage:.1f}%", "Missing Data", "#ef4444" if qr.missing_percentage > 20 else "#10b981"),
            (f"{qr.exact_duplicate_rows:,}", "Exact Duplicates", "#f59e0b" if qr.exact_duplicate_rows > 0 else "#10b981")
        ]
        
        for col, (value, label, color) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {color};">
                    <div class="metric-value" style="color: {color};">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional Metrics Row
        st.markdown("### 📈 Additional Insights")
        cols2 = st.columns(3)
        
        with cols2[0]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #f59e0b;">{qr.fuzzy_duplicate_groups}</div>
                <div class="metric-label">Fuzzy Duplicate Groups</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols2[1]:
            total_cells = qr.total_cells
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #3b82f6;">{total_cells:,}</div>
                <div class="metric-label">Total Cells</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols2[2]:
            issues_count = len(qr.columns_with_issues)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {'#ef4444' if issues_count > 0 else '#10b981'};">{issues_count}</div>
                <div class="metric-label">Columns with Issues</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Insights Section
        st.markdown("### 💡 Data Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            # Data Health
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">🏥 Data Health Status</div>', unsafe_allow_html=True)
            
            health_items = []
            if qr.missing_percentage < 5:
                health_items.append("✅ Low missing data (< 5%)")
            elif qr.missing_percentage < 20:
                health_items.append("⚠️ Moderate missing data (5-20%)")
            else:
                health_items.append("🔴 High missing data (> 20%)")
            
            if qr.exact_duplicate_rows == 0:
                health_items.append("✅ No exact duplicates found")
            else:
                health_items.append(f"⚠️ {qr.exact_duplicate_rows:,} duplicate rows detected")
            
            if qr.fuzzy_duplicate_groups > 0:
                health_items.append(f"🔍 {qr.fuzzy_duplicate_groups} fuzzy groups found")
            else:
                health_items.append("✅ No fuzzy duplicates detected")
            
            for item in health_items:
                st.write(item)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Column Issues
            if qr.columns_with_issues:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-header">⚠️ Columns Needing Attention ({len(qr.columns_with_issues)})</div>', unsafe_allow_html=True)
                
                for col in qr.columns_with_issues[:5]:
                    if col in st.session_state.column_profiles:
                        profile = st.session_state.column_profiles[col]
                        issues = []
                        if profile.null_count > 0:
                            issues.append(f"{profile.null_percentage:.1f}% null")
                        if profile.special_chars:
                            issues.append("special chars")
                        if profile.outliers["count"] > 0:
                            issues.append("outliers")
                        if issues:
                            st.markdown(f'<div class="warning-card"><b>{col}</b>: {", ".join(issues)}</div>', unsafe_allow_html=True)
                
                if len(qr.columns_with_issues) > 5:
                    st.caption(f"... and {len(qr.columns_with_issues) - 5} more")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with insights_col2:
            # Data Distribution
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📈 Column Distribution</div>', unsafe_allow_html=True)
            
            numeric_cols = [c for c, p in st.session_state.column_profiles.items() 
                           if 'int' in p.dtype or 'float' in p.dtype]
            
            if numeric_cols:
                selected = st.selectbox("Select numeric column:", numeric_cols, key="dist_select")
                if selected in st.session_state.df.columns:
                    data = st.session_state.df[selected].dropna()
                    if len(data) > 0:
                        st.bar_chart(data.value_counts().head(10))
                    else:
                        st.info("No data available for this column")
            else:
                st.info("No numeric columns available for visualization")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent Operations
            if st.session_state.fixes_applied:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-header">📝 Recent Operations</div>', unsafe_allow_html=True)
                st.markdown('<div class="operation-log">', unsafe_allow_html=True)
                
                for fix in reversed(st.session_state.fixes_applied[-5:]):
                    st.markdown(f"""
                    <div class="operation-item">
                        <span class="timestamp">{fix.get('timestamp', 'N/A')}</span> - {fix.get('operation', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Error rendering dashboard: {str(e)}</div>', unsafe_allow_html=True)
        st.error("Please try refreshing the page or re-uploading your data file.")