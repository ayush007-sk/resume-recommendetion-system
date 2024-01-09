"""
importing all the libraries need 
"""
import os
import pickle
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt

import predict_data as prd

st.set_page_config(layout="wide")
# importing a css file
with open("style.css") as f:
    st.markdown('''<style>{}</style>'''.format(f.read()), unsafe_allow_html=True)

CATEGORI=list(pickle.load(open("categories.pkl","rb")))
CATEGORI.insert(0,"All")

PATH="test_resume"
# Title of the web app
st.title("Resume Recommendation System")

# creat a form 
with st.form(key='my_form'):
    st.text("Feed all you Resumes in your resumes directory, I will recommend you best candidates for any job role .")
    
    option = st.selectbox(
    'Select the Job Profile : ',
    tuple(CATEGORI))  
        

    st.markdown('<style>div.row-widget.stButton > button {background-color: red;}</style>', unsafe_allow_html=True)
    submit_button = st.form_submit_button(label='Submit') 
    
    if submit_button:
        outputs,uid=prd.test_data(PATH)
        if option =="All":
            output=prd.filter_data(outputs,uid)
        else:
            output=prd.filter_data(outputs,uid,option)
        if output is not None:
                st.dataframe(output.sort_values(["confidence_r1"],ignore_index=True))
        else:
            st.write("No Match found!")
st.title("Check how much suitable your resume is for the job profile")
with st.form(key="my_form1"):
    file=st.file_uploader("Upload the resume here..",accept_multiple_files=False,type=["pdf"])

    submitted=st.form_submit_button(label="Upload")

    if submitted:
        try:
            os.mkdir("temp")
        except:
            pass
        with open("temp/uploaded.pdf", "wb") as f:
            f.write(file.getvalue())
        outputs,uid=prd.test_data("temp")
        output=prd.filter_data(outputs,uid)
        col1,col2=st.columns(2)
        with col1:
            st.dataframe(output)
        with col2:
            data={}
            for i in range(3):
                data[output[f"role_{i+1}"][0]]=float(output[f"confidence_r{i+1}"][0].replace("%",""))
            data["rest"]=100-sum(data.values())
            fig,ax=plt.subplots()
            ax.pie(data.values(),labels=data.keys(),autopct="%.1f%%")
            st.pyplot(fig,transparent=True)
        