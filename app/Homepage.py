import streamlit as st

st.set_page_config(
    page_title="PhD Work",
    page_icon="👋",
)

st.write("# Welcome to Vivek's PhD! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    This is a StreamLit app that showcases my PhD work on Quantum Approximate Optimization Algorithm (QAOA).

    The app is divided into three sections:
    - **QAOA Landscape**: Here we visualize the QAOA landscape at different layers for various instance classes
    - **QAOA Parameter Evolution**: On this page we visualize the evolution of QAOA parameters -- in particular we're interested in how the parameters concentrate around different modes.
    - **Instance Space Analysis**: We provide an interface to conduct Instance Space Analysis for various experiments studying the QAOA algorithm
    """
)