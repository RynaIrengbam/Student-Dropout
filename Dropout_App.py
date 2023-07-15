import pandas as pd
import streamlit as st
import pickle
import numpy as np

#Title of the web app 
st.header("School Dropout Prediction")

def user_input_features():
    Residence_city = st.selectbox('Residence City', ('LOCAL', 'NEIGHBOR', 'FOREIGN'))
    State = st.selectbox('State', ('LOCAL', 'NEIGHBOR', 'FOREIGN'))
    Province = st.selectbox('Province', ('LOCAL', 'NEIGHBOR', 'FOREIGN'))
    Socioeconomic_level = st.selectbox('Socioeconomic Level', (1,2,3,4,5,6))
    Vulnerable_group = st.selectbox('Vulnerable Group', (-1,1,2))
    Civil_status = st.selectbox('Civil Status', ('Married', 'Unmarried', 'Free union', 'Separated', 'Single'))
    Age = st.number_input('Age')
    Family_income = st.number_input('Family income')
    Father_level = st.selectbox('Father Education Level', ("UNDERGRADUATE","HIGH SCHOOL","PRIMARY SCHOOL","TECHNICAL","TECHNOLOGIST","UNREGISTERED"))
    Mother_level = st.selectbox('Mother Education Level', ("UNDERGRADUATE","HIGH SCHOOL","PRIMARY SCHOOL","TECHNICAL","TECHNOLOGIST","UNREGISTERED"))
    Desired_program = st.selectbox('Desired Program', ('UNSPECIFIED', 'INFORMATIC ENGINEERING', 'ELECTRIC AUTOMATION TECHNOLOGY'))
    STEM_subjects = st.number_input('STEM subject Grade')
    H_subjects = st.number_input('H subject Grade')

    my_dict = {
        "Residence_city": Residence_city,
        "Socioeconomic_level": Socioeconomic_level,
        "Civil_status": Civil_status,
        "Age": Age,
        "State": State,
        "Province": Province,
        "Vulnerable_group": Vulnerable_group,
        "Desired_program": Desired_program,
        "Family_income": Family_income,
        "Father_level": Father_level,
        "Mother_level": Mother_level,
        "STEM_subjects": STEM_subjects,
        "H_subjects": H_subjects
                  }
    features = pd.DataFrame(my_dict, index=[0])
    return features

input_df = user_input_features()

data1 = pd.read_csv("data/Merged.csv")
data1.fillna(0, inplace=True)
data = data1.drop(columns = 'Dropout')

df = pd.concat([input_df,data],axis=0)

encode = ['Residence_city', 'Civil_status', 'State',
       'Province', 'Desired_program',
       'Father_level', 'Mother_level']
       
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]
df.fillna(0, inplace = True)

features1 = ['Socioeconomic_level', 'Age', 'Vulnerable_group', 'Family_income',
       'STEM_subjects', 'H_subjects', 'Residence_city_LOCAL',
       'Residence_city_NEIGHBOR', 'Civil_status_Married',
       'Civil_status_Separated', 'Civil_status_Single',
       'Civil_status_Unmarried', 'State_LOCAL', 'State_NEIGHBOR',
       'Province_LOCAL', 'Province_NEIGHBOR',
       'Desired_program_INFORMATIC ENGINEERING', 'Desired_program_UNSPECIFIED',
       'Father_level_PRIMARY SCHOOL', 'Father_level_TECHNICAL',
       'Father_level_TECHNOLOGIST', 'Father_level_UNDERGRADUATE',
       'Father_level_UNREGISTERED', 'Mother_level_PRIMARY SCHOOL',
       'Mother_level_TECHNICAL', 'Mother_level_TECHNOLOGIST',
       'Mother_level_UNDERGRADUATE', 'Mother_level_UNREGISTERED']
df = df[features1]



#loading the saved model 
loaded_model = pickle.load(open('C:/My Stuff/Internship/Resolute AI/Data/Student dropout/finalized_model3.sav', 'rb'))

#creating the button
prediction = loaded_model.predict(df)
if st.button("Prediction"):
    if prediction == 0:
        prediction = "Will not Dropout"
    else:
        prediction = "Will Dropout"
    st.info(prediction)
