import streamlit as st

def history_page():
    st.title("History")
    if 'history' not in st.session_state or not st.session_state['history']:
        st.write("No history available.")
    else:
        st.subheader("Conversation History")
        for idx, (q, ans) in enumerate(st.session_state['history'], start=1):
            st.markdown(f"**Q{idx}:** {q}", unsafe_allow_html=True)
            st.markdown(f"**A{idx}:** {ans}", unsafe_allow_html=True)

history_page()
