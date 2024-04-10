import streamlit as st

st.set_page_config(
    page_title="PhD Work",
    page_icon="👋",
)

st.write("# Welcome to Vivek's PhD! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This is a StreamLit app that showcases my PhD work.

    The app is divided into two sections:
    - **Instance Space Analysis**: This section provides an overview of the instance space analysis for the problem.
    - **Landscape**: This section provides an overview of the optimal angles for the problem.
    
    $$ f(x) = x^2 $$
"""
)