import streamlit as st
from .logic import authenticate

def render_login_screen():
    """Render the login form"""
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: auto;
            padding: 2rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<p style='text-align: center; color: #666;'>Enter your credentials to access the data quality platform</p>", unsafe_allow_html=True)
        
        # Centering form in a column
        _, col, _ = st.columns([1, 2, 1])
        
        with col:
            username = st.text_input("Username", placeholder="Example")
            password = st.text_input("Password", type="password", placeholder="Pass****")
            
            # hide login button if username and password is empty conditionaly
            if username and password:
                submit = st.button("Login", use_container_width=True, type="primary")
                
                if submit:
                    user_info = authenticate(username, password)
                    if user_info:
                        st.session_state.app_state.authenticated = True
                        st.session_state.app_state.user_name = user_info['name']
                        st.session_state.app_state.username = user_info['username']
                        
                        from state.session import _save_persisted_data
                        _save_persisted_data()
                        
                        st.success(f"Welcome back, {user_info['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                        # make image in center
                        _, col_img, _ = st.columns([1, 2, 1])
                        with col_img:
                            st.caption("If you wanted to access this tool kindly pay and get your credentials, for trial one day 5000, for one month 10000, for one year 50000")
                            st.image("assets/Images/QR.jpg", caption="Scan to Pay", use_container_width=True)

        # st.markdown("---")
       
