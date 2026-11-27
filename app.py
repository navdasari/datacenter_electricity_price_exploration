"""Streamlit dashboard: what moved residential electricity prices, and how much was data centers.

Run:  streamlit run app.py

Pages
  Results                     the published baseline (views/results.py; reads output/)
  Assumptions & sensitivity   change assumptions and compare (views/assumptions.py; runs src/pipeline.py)
"""
import streamlit as st

st.set_page_config(page_title="Data Centers & Power Prices", layout="wide")

page = st.navigation([
    st.Page("views/results.py", title="Results", default=True),
    st.Page("views/assumptions.py", title="Assumptions & sensitivity", url_path="assumptions"),
])
page.run()
