import streamlit as st
import pandas as pd
import numpy as np
#import joblib
#from churnback2 import columns
from sklearn.preprocessing import MinMaxScaler
#from churnback2 import project
from z2 import columns
project = pd.DataFrame([np.array(columns)] ,columns=columns )

gende = st.selectbox("input sex" , ['Female','Male'])
seniorcitizen = st.selectbox("are u SeniorCitizen ?" , ['Yes','No'])
partner = st.selectbox("r u have Partner ? or she left u ?" ,['Yes','No'])
dependents = st.selectbox("r u have Dependents?",['Yes','No'])
tenure = st.slider("tenure",0,80)
phoneservice = st.selectbox("r u have PhoneService ?" ,['Yes','No'])
multiplelines = st.selectbox("r u MultipleLines ? " ,['Yes','No'])
internetservice = st.selectbox("choise your InternetService ?",['DSL', 'Fiber optic' ,'No'])
onlinesecurity = st.selectbox('OnlineSecurity ?' , ['Yes','No'])
onlinebackup = st.selectbox("OnlineBackup ?", ['Yes','No'] )
deviceprotection = st.selectbox("DeviceProtection?", ['Yes','No'])
techSupport = st.selectbox("TechSupport?", ['Yes','No'])
streamingTV = st.selectbox("StreamingTV?", ['Yes','No'])
streamingMovies = st.selectbox("StreamingMovies?", ['Yes','No'])
contract = st.selectbox("Wen will u renew  contract?", ['Month-to-month', 'One year', 'Two year'])
paperlessBilling = st.selectbox("PaperlessBilling?", ['Yes','No'])
paymentMethod = st.selectbox("your PaymentMethod :" ,['Electronic check' ,'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])
monthlycharges = st.slider("MonthlyCharges",0,120)
#totalcharges = st.slider("TotalCharges",0,10000)
totalcharges = st.number_input("TotalCharges")
def predict():
    
    #project = pd.DataFrame([np.array(columns)] ,columns=columns )

    #from churnback3 import everything

    #project.drop(columns='Churn' , axis=1 , inplace= True)
    
    #project.iloc[: , 0]  = gende
    project.loc[:,['gender']] = gende
    #project.iloc[: , 1]  = seniorcitizen
    project.loc[:,['SeniorCitizen']] = seniorcitizen
    #project.iloc[: , 2]  = partner
    project.loc[:,['Partner']] = partner
    #project.iloc[: , 3]  = dependents
    project.loc[:,['Dependents']] = dependents
    #project.iloc[: , 4]  = tenure
    project.loc[:,['tenure']] = tenure
    #project.iloc[: , 5]  = phoneservice
    project.loc[:,['PhoneService']] = phoneservice
    #project.iloc[: , 6]  = multiplelines
    project.loc[:,['MultipleLines']] = multiplelines
    #project.iloc[: , 7]  = internetservice
    if internetservice == 'DSL' :
        project.loc[: ,['InternetService_DSL'] ] = 1
        project.loc[: ,['InternetService_Fiber optic'] ] = 0
        project.loc[: ,['InternetService_No'] ] = 0
    elif internetservice == 'Fiber optic' :
        project.loc[: ,['InternetService_DSL'] ] = 0
        project.loc[: ,['InternetService_Fiber optic'] ] = 1
        project.loc[: ,['InternetService_No'] ] = 0  
    else :
        project.loc[: ,['InternetService_DSL'] ] = 0
        project.loc[: ,['InternetService_Fiber optic'] ] = 0
        project.loc[: ,['InternetService_No'] ] = 1
                     
    project.loc[: , ['OnlineSecurity']]  = onlinesecurity
    project.loc[: , ['OnlineBackup']]  = onlinebackup
    project.loc[: , ['DeviceProtection']] = deviceprotection
    project.loc[: , ['TechSupport']] = techSupport
    project.loc[: , ['StreamingTV']] = streamingTV
    project.loc[: , ['StreamingMovies']] = streamingMovies
    project.loc[: , ['PaperlessBilling']] = paperlessBilling
    #project.iloc[: , 14] = contract

    if contract == 'Month-to-month' :   
        project.loc[: , ['Contract_One year'] ] = 0
        project.loc[: , ['Contract_Two year'] ] = 0
        project.loc[: , ['Contract_Month-to-month'] ] = 1
    elif contract == 'One year' :
        project.loc[: , ['Contract_One year'] ] = 1
        project.loc[: , ['Contract_Two year'] ] = 0
        project.loc[: , ['Contract_Month-to-month'] ] = 0 
    else :
        project.loc[: , ['Contract_One year'] ] = 0
        project.loc[: , ['Contract_Two year'] ] = 1
        project.loc[: , ['Contract_Month-to-month'] ] = 0 


    if paymentMethod == 'Electronic check' :
        project.loc[: , ['PaymentMethod_Electronic check'] ] = 1
        project.loc[: , ['PaymentMethod_Bank transfer (automatic)'] ] = 0
        project.loc[: , ['PaymentMethod_Credit card (automatic)'] ] = 0
        project.loc[: , ['PaymentMethod_Mailed check'] ] = 0
    elif paymentMethod == 'Mailed check' :  
        project.loc[: , ['PaymentMethod_Electronic check'] ] = 0
        project.loc[: , ['PaymentMethod_Bank transfer (automatic)'] ] = 0
        project.loc[: , ['PaymentMethod_Credit card (automatic)'] ] = 0
        project.loc[: , ['PaymentMethod_Mailed check'] ] = 1
    elif  paymentMethod == 'Bank transfer (automatic)' : 
        project.loc[: , ['PaymentMethod_Electronic check'] ] = 0
        project.loc[: , ['PaymentMethod_Bank transfer (automatic)'] ] = 1
        project.loc[: , ['PaymentMethod_Credit card (automatic)'] ] = 0
        project.loc[: , ['PaymentMethod_Mailed check'] ] = 0 
    else :
        project.loc[: , ['PaymentMethod_Electronic check'] ] = 0
        project.loc[: , ['PaymentMethod_Bank transfer (automatic)'] ] = 0
        project.loc[: , ['PaymentMethod_Credit card (automatic)'] ] = 1
        project.loc[: , ['PaymentMethod_Mailed check'] ] = 0 

    #project.iloc[: , 20] = paymentMethod
    project.loc[: , ['MonthlyCharges']] = monthlycharges
    project.loc[: , ['TotalCharges']] = totalcharges
    
    yes_no_columns = ['SeniorCitizen' ,'Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yes_no_columns :
        project[col].replace({'Yes':1 , 'No':0} , inplace=True)
    project['gender'].replace({'Female':1 , 'Male' : 0} , inplace=True)

    cals_to_scale = ['tenure','MonthlyCharges','TotalCharges']
    scaler = MinMaxScaler()
    project[cals_to_scale] = scaler.fit_transform(project[cals_to_scale])    
    #pred = joblib.load('logreg.joblib')
    #st.dataframe(project)
    #st.success(everything(project))
    from churnback4 import model_fited
    #from churnback5 import projectlast
    #st.success( model_fited.predict(projectlast) )
    st.success(f"CHUrn : {model_fited.predict(project)[0]}")

st.button("calculate" , on_click= predict)