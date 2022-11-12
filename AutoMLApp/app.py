import streamlit as st
import pandas as pd
#from pandas_profiling import pro
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report 
from pycaret.classification import setup,compare_models,save_model,pull

if os.path.exists("sourcedata.csv"):
    df=pd.read_csv("sourcedata.csv",index_col=None)



with st.sidebar:
    st.image("goku.jpeg")
    st.title("GOKUS AutoML")
    choices=st.radio("navigation",["Upload","Profiling","ML","Download"])
if choices=="Upload":
    file=st.file_uploader("Upload you Data for Modelling")
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)

if choices=="Profiling":
    st.title("Exploratory Data Analysis")
    profile_report=df.profile_report()
    st_profile_report(profile_report)
    
if choices=="ML":
    st.title("ML Part")
    target=st.selectbox("Select ur target",df.columns)
    if st.button("Train Model"):
        setup(df,target=target,silent=True)
        setup_df=pull()
        st.info("This is ML experiment Settings")
        st.dataframe(setup_df)
        best_model=compare_models()
        compare_df=pull()
        st.info("This is the ML MOdel")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model")
if choices=="Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("Download File",f,"trained_model.pkl")       
    