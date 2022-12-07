import streamlit as st
import pandas as pd
import time
import os
def app():
    st.title('APLIKASI SENTIMEN ANALASIS')
    data = st.file_uploader("upload data berformat csv", type=['csv'])
    
    if data is not None:
        try :
            dataframe = pd.read_csv(data,lineterminator='\n')
            st.write(dataframe)

            col1, col2 = st.columns(2)
            with col1 :
                column = st.selectbox("Pilih Kolom yang akan di proses :",
                list(dataframe.columns))
            with col2 :
                label = st.selectbox("Pilih Kolom yang akan dijadikan label atau class :",
                list(dataframe.columns))
  
            column_data = pd.DataFrame(data={'column': [column], 'label': [label]})
            if st.button('simpan data') :
                column_data.to_csv('data/meta/column_data.csv',index=False)
                if os.path.exists("data/data_branch.csv"):
                    os.remove("data/data_branch.csv")
                if os.path.exists("data/tf_idf.csv"):
                    os.remove("data/tf_idf.csv")
                dataframe = dataframe[[column_data['column'][0],column_data['label'][0]]]
                dataframe.to_csv('data/main_data.csv',index=False)
                with st.spinner('tunggu sebentar ...'):
                    time.sleep(1)
                st.success('data berhasil disimpan')
                st.info('column ' + column_data['column'][0] + ' akan diproses')
                st.info('column ' + column_data['label'][0] + ' akan dijadikan label')
        except :
            st.error('error : periksa lagi inputan anda')
