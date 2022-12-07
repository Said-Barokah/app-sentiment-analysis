import streamlit as st
import pandas as pd
import timeit
import time
def app():
    from googletrans import Translator,LANGUAGES
    # from google_trans_new import google_translator,LANGUAGES
    translator = Translator()
    # translator = google_translator()
    data = pd.read_csv('data/main_data.csv')
    column_data = pd.read_csv('data/meta/column_data.csv')
    column = column_data['column'][0]
    label = column_data['label'][0]
    key_list_leng = list(LANGUAGES.keys())
    val_list_leng = list(LANGUAGES.values())
    col_trans1, col_trans2 = st.columns(2)
    with col_trans1:
        st.subheader('Data Sekarang')
        from_trans = st.selectbox("translate data dari bahasa :",
            val_list_leng,index=43,key='from_trans')
        key_from_leng = key_list_leng[val_list_leng.index(from_trans)]
    with col_trans2:
        st.subheader('Data setelah di Translate')
        to_trans = st.selectbox("translate data ke bahasa :",
            ['arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', 'indonesian','norwegian', 'portuguese', 'romanian', 'russian', 'spanish', 'swedish'],index=3,key='to_trans')
        key_to_leng = key_list_leng[val_list_leng.index(to_trans)]
    col1, col2 = st.columns(2)
    with col1 :
        st.dataframe(data)
    with col2 :
        with st.spinner('tunggu sebentar ...'):
            start = timeit.default_timer()
            data[column] = data[column].apply(translator.translate, src=key_from_leng, dest= key_to_leng)
            data[column] = data[column].apply(getattr, args=(column,))
            # data[column] = data[column].apply(translator.translate,  lang_src=key_from_leng, lang_tgt= key_to_leng)
            # translator.translate()
            stop = timeit.default_timer()
            data.to_csv('data/data_branch.csv',index=False)
            data_branch = pd.read_csv('data/data_branch.csv')
            st.dataframe(data)
            st.write('proses translate : ', stop-start,' detik')
            
            if (st.button('simpan data')):
                data_branch.to_csv('data/main_data.csv',index=False)
                with st.spinner('tunggu sebentar ...'):
                    time.sleep(1)
                st.success('Berhasil disimpan')
    
