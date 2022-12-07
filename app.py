import streamlit as st
import pandas as pd
from halaman import upload_data,pre_processing,translator, fea_extraction, classification
page_names_to_funcs = {
    "Upload Data" : upload_data.app,
    "Translator"  : translator.app,
    "Pre Processing" : pre_processing.app,
    "Feature Extraction" : fea_extraction.app,
    "Classification" : classification.app
}

demo_name = st.sidebar.selectbox("halaman", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()