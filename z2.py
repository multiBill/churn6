import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#import warnings
#warnings.filterwarnings('ignore')
#from churnconnect import project

df = pd.read_csv(r"z3_IT_customer_churn.csv")
#columns = df.columns
#from churnconnect import project
#project = pd.DataFrame([np.array(columns)] ,columns=columns )
#df = project
#def everything(df):
df1 = df[df['TotalCharges']!=' ']
df1['TotalCharges'] = pd.to_numeric(df1.TotalCharges)

df1.replace('No phone service','No' , inplace=True)
df1.replace('No internet service' , 'No' , inplace=True)

yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns :
   df1[col].replace({'Yes':1 , 'No':0} , inplace=True)

df1['gender'].replace({'Female':1 , 'Male' : 0} , inplace=True)

live = pd.get_dummies(df1[['PaymentMethod','Contract','InternetService']],dtype=int)
df1 = df1.merge(live ,left_index=True,right_index=True )
df1.drop(columns=['PaymentMethod','Contract','InternetService'],inplace=True)

cals_to_scale = ['tenure','MonthlyCharges','TotalCharges']
scaler = MinMaxScaler()
df1[cals_to_scale] = scaler.fit_transform(df1[cals_to_scale])

#x = df1.drop(columns='Churn')
#y = df1.Churn.astype(np.float32)
#smote = SMOTE(sampling_strategy='minority')
#x_sm , y_sm = smote.fit_resample(x,y)
columns = df1.drop('Churn',axis='columns').columns

count_class_0, count_class_1 = df1.Churn.value_counts()
df_class_0 = df1[df1['Churn'] == 0]
df_class_1 = df1[df1['Churn'] == 1]
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)
x_sm = df_test_over.drop('Churn',axis='columns')
y_sm = df_test_over['Churn']

x_train,x_test,y_train,y_test = train_test_split(x_sm,y_sm,random_state=25,test_size=0.2 , stratify=y_sm)




def logreg(x_train,y_train,x_test,y_test ,weights) :
   if weights == -1 :
      model = LogisticRegression()
   else :
      model = LogisticRegression( class_weight={ 0 : weights[0] , 1 : weights[1] } )  
   model.fit(x_train,y_train)
   #y_pred = model.predict(x_test)
   #acc_train = model.score(x_train,y_train)
   #acc_test = model.score(x_test,y_test)
   #cl_rep = classification_report(y_test,y_pred)
   return model

weights = [1,1.5] # pass -1 to use Logistics Regression without weights
model_fited = logreg(x_train, y_train, x_test, y_test, weights) 
#from churnback5 import projectlast
#result = logreg(x_train, y_train, projectlast, weights)   
#st.success(result)
