#importing necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
import pickle
#---------------------------------------------------------------------------------------------------------------------
#Streamlit page Configurartion
st.set_page_config(
    page_title="Industrial Copper Data Prediction ",
    page_icon=":chart_with_upwards_trend:", 
    layout="wide",  
)

st.info("### :factory: &nbsp; INDUSTRIAL COPPER DATA PREDICTOR ")
st.markdown("Choose a Tab for Prediction:")
tab1, tab2 = st.tabs([":orange[SELLING PRICE :money_with_wings:]", ":orange[STATUS:triangular_flag_on_post:]"])
#---------------------------------------------------------------------------------------------------------------
#Selling price prediction tab

with tab1: 
        item_type_   =  ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_    =   [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_ =  [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product_     =  ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                         '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                         '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                         '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                         '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
        status_     =  ['Won', 'Lost','Draft', 'To be approved', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']

        with st.form("form_1"):
            col1,col2=st.columns([5,5])
            with col1:
                item_type = st.selectbox("Item Type", item_type_,key=2)
                country = st.selectbox("Country", sorted(country_),key=3)
                application = st.selectbox("Application", sorted(application_),key=4)
                product_ref = st.selectbox("Product Reference", product_,key=5)
                status = st.selectbox("Status", status_,key=1)
            with col2:               
                st.write(' ')
                customer_1 = st.text_input("Enter Customer ID (Min:12458 & Max:30408185)")
                quantity_1 = st.text_input("Enter Quantity")
                thickness_1 = st.text_input("Enter Thickness (Min:0.18 & Max:400)")
                width_1= st.text_input("Enter Width (Min:1 & Max:2990)")
                st.markdown(" :blue[Press the Below Button :]")
                submit_button_1= st.form_submit_button(label=" :red[PREDICT SELLING PRICE]")
            #Input pattern matching using regular expression
            success_1=1
            pattern_1= "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_1,thickness_1,width_1,customer_1]:             
                if re.match(pattern_1, i):
                     pass
                else:                    
                     success_1=0
                     break                  
        if submit_button_1 and success_1==0:
                    st.text("Warning:")
                    st.info(" :x: Kindly Check your Entry!!!")       
        if submit_button_1 and success_1==1:
                    #Opening models from saved file
                    with open(r'C:\Users\banup\Desktop\Industrial_copper_Modellimg\model.pkl', 'rb') as file_1:
                        dtr_model = pickle.load(file_1)
                    with open(r'C:\Users\banup\Desktop\Industrial_copper_Modellimg\sc.pkl', 'rb') as file_2:
                        dtr_sc = pickle.load(file_2)
                    with open(r'C:\Users\banup\Desktop\Industrial_copper_Modellimg\encoder_1.pkl', 'rb') as file_2:
                        dtr_encoder_1= pickle.load(file_2)
                    with open(r'C:\Users\banup\Desktop\Industrial_copper_Modellimg\encoder_2.pkl', 'rb') as file_2:
                        dtr_encoder_2= pickle.load(file_2) 
                    #Fitting inputs with the model
                    new_data_1= np.array([[np.log(float(quantity_1)),application,np.log(float(thickness_1)),float(width_1),country,float(customer_1),int(product_ref),item_type,status]])
                    new_data_1_encoder_1 = dtr_encoder_1.transform(new_data_1[:, [7]]).toarray()
                    new_data_1_encoder_2 = dtr_encoder_2.transform(new_data_1[:, [8]]).toarray()
                    new_data_1 = np.concatenate((new_data_1[:, [0,1,2, 3, 4, 5, 6,]], new_data_1_encoder_1, new_data_1_encoder_2 ), axis=1)
                    new_data_1 = dtr_sc.transform(new_data_1)
                    #Predicting the output
                    prediction_1 = dtr_model.predict(new_data_1) 
                    predicted_price=np.exp(prediction_1)
                    st.markdown(f"##  :dollar: :green[Predicted Selling Price:] **${predicted_price[0]:,.2f}**")
#---------------------------------------------------------------------------------------------------------------
 #Status Prediction Tab 
                                      
with tab2:
        with st.form("form_2"):                   
                col1,col2=st.columns([5,5])
                with col1:
                        st.write(' ')
                        quantity_2 = st.text_input("Enter Quantity")
                        thickness_2 = st.text_input("Enter Thickness (Min:0.18 & Max:400)")
                        width_2 = st.text_input("Enter  Width (Min:1 & Max:2990)")
                        customer_2= st.text_input("Customer ID (Min:12458 $ Max:30408185)")
                        selling_price_2= st.text_input("Selling Price (Min:1 $ Max:100001015)") 
                    
                with col2:    
                        st.write(' ')
                        item_type_2 = st.selectbox("Item Type", item_type_,key=21)
                        country_2 = st.selectbox("Country", sorted(country_),key=31)
                        application_2 = st.selectbox("Application", sorted(application_),key=41)  
                        product_ref_2 = st.selectbox("Product Reference", product_,key=51)           
                        st.markdown(" :blue[Press the Below Button :]")
                        submit_button_2= st.form_submit_button(label=":red[PREDICT STATUS]")

                success_2=1
                pattern_2 = "^(?:\d+|\d*\.\d+)$"
                for i in [quantity_2,thickness_2,width_2,customer_2,selling_price_2]:             
                      if re.match(pattern_2, i):
                          pass
                      else:                    
                         success_2=0
                         break     
                                   
        if submit_button_2 and success_2==0:
                st.text("Warning:")
                st.info(" :x: Kindly Check your Entry!!!")  
        if submit_button_2 and success_2==1:
                    #Opening models from saved file  
                    with open(r"C:\Users\banup\Desktop\Industrial_copper_Modellimg\model_2_class.pkl", 'rb') as file_1:
                          dtr_model = pickle.load(file_1)
                    with open(r'C:\Users\banup\Desktop\Industrial_copper_Modellimg\sc_2.pkl', 'rb') as file_2:
                          dtr_sc = pickle.load(file_2)
                    with open(r"C:\Users\banup\Desktop\Industrial_copper_Modellimg\encoder_3.pkl", 'rb') as file_2:
                          dtr_encoder_3 = pickle.load(file_2)
                    #Fitting inputs with the model
                    new_data_2 = np.array([[np.log(float(quantity_2)), np.log(float(selling_price_2)), application_2, np.log(float(thickness_2)),float(width_2),country_2,int(customer_2),int(product_ref_2),item_type_2]])
                    new_data_encoder_3 = dtr_encoder_3 .transform(new_data_2 [:, [8]]).toarray()
                    new_data_2  = np.concatenate((new_data_2 [:, [0,1,2, 3, 4, 5, 6,7]], new_data_encoder_3), axis=1)
                    new_data_2  = dtr_sc.transform(new_data_2)
                    #Predicting the output
                    prediction_2  = dtr_model.predict(new_data_2)
                    predicted_status=np.exp(prediction_2)
                    if predicted_status==1:
                        st.markdown('## :white[Status:]:green[Won]:thumbsup: ')
                    else:
                        st.markdown('## :white[Status :]:red[LOST]:thumbsdown: ')
st.markdown("### ")
st.write('Modelled by Banuprakash Vellingiri &nbsp; :heart:')
#---------------------------------------------------------------------------------------------------------------
