import streamlit as st
import pandas as pd
import pickle
import os
#import sklearn

current_dir = os.getcwd()


# Construct the relative path to the file

# image_path = os.path.join(current_dir,r"dataset\car_image.png")
# csv_path = os.path.join(current_dir,r"dataset\cleaned_car_price_csv.csv")
# model_path_a = os.path.join(current_dir,r"code\saved_model_for_car_price.pkl")

image_path = os.path.join(current_dir,"car_image.png")
csv_path = os.path.join(current_dir,"cleaned_car_price_csv.csv")
model_path_a = os.path.join(current_dir,"saved_model_for_car_price.pkl")
print("-"*36)
print("model path is :",model_path_a)
print("csv path is ", csv_path)
print("-"*36)
print("\n")

car_df= pd.read_csv(csv_path)

try:
    with open(model_path_a, 'rb') as model_file:
        model = pickle.load(model_file)
    try:
        print(model)
    except:
        print("Model is not printed")
except Exception as e:
    print("Error loading the model:", e)

st.set_page_config(page_title="Car Price Predicter",page_icon="🚗",layout="centered")

st.markdown("<div style='background-color:#219C90; border-radius:50px; align-items:center; justify-content: center;'><h1 style='text-align:center; color:white;'>Car Price Predictor</h1></div>",unsafe_allow_html=True)

st.markdown("<h4 style='text-align:center; color:black;'>Calculate The Best Price For The Car</h4>",unsafe_allow_html=True)

#Styling Streamlit Web App
col1 , col2 = st.columns(2)

with col1:
    st.write("")
    st.image(image=image_path,use_column_width=True,caption="Here comes Car Price Predictor")


with col2:
    car_model = st.selectbox(label="Select The car model",options=car_df['Car Model'].unique(),placeholder="Select",index=None)
    kms_driven = st.number_input(label="Enter The Kms Driven",min_value=1500,max_value=150000,value=None)
    col3,col4 = st.columns(2)
    
    with col3:
        fuel_type = st.radio(label="Select the Fuel Type",options=car_df["Fuel Type"].unique(),index=None)
    with col4:
        suspension_type = st.radio(label="Select the Suspension Type",options=car_df["Suspension"].unique(),index=None)

    year = st.slider(min_value=2000,max_value=2024,label="Select The Year of Car")
    pred = st.button("Predict",use_container_width=True)

data = {'Year': [year], 'kms Driven': [kms_driven]}
df1 = pd.DataFrame(data=data)

dummies_fuel = pd.get_dummies(car_df["Fuel Type"],dtype = "int",drop_first=True)
dummies_suspension = pd.get_dummies(car_df["Suspension"],dtype = "int",drop_first=True)
dummies_model = pd.get_dummies(car_df["Car Model"],dtype = "int",drop_first=True)

car_df.drop(columns=["Fuel Type","Suspension","Car Model"],axis = "columns",inplace= True)
concated_df = pd.concat([car_df,dummies_fuel,dummies_suspension,dummies_model], axis="columns")
#print(concated_df.columns.to_list)

model_features = ['Year', 'kms Driven', 'Diesel', 'Petrol', 'Manual',
       'Honda Accord VTi-L', 'Honda Amaze E', 'Honda Amaze EX',
       'Honda Amaze Exclusive', 'Honda Amaze S', 'Honda Amaze SX',
       'Honda Amaze V', 'Honda Amaze VX', 'Honda Amaze i-DTEC',
       'Honda BR-V i-DTEC', 'Honda BR-V i-VTEC', 'Honda Brio 1.2',
       'Honda Brio EX', 'Honda Brio S', 'Honda Brio V', 'Honda Brio VX',
       'Honda CR-V 2.0', 'Honda CR-V 2.0L', 'Honda CR-V 2.4',
       'Honda CR-V 2.4L', 'Honda CR-V RVi', 'Honda City 1.3', 'Honda City 1.5',
       'Honda City Anniversary', 'Honda City Corporate', 'Honda City E',
       'Honda City EXi', 'Honda City GXi', 'Honda City S', 'Honda City SV',
       'Honda City V', 'Honda City VTEC', 'Honda City VX', 'Honda City ZX',
       'Honda City i', 'Honda City i-DTEC', 'Honda City i-VTEC',
       'Honda Civic 1.8', 'Honda Jazz 1.2', 'Honda Jazz 1.5',
       'Honda Jazz Basic', 'Honda Jazz S', 'Honda Jazz Select', 'Honda Jazz V',
       'Honda Jazz VX', 'Honda Jazz X', 'Honda Mobilio E', 'Honda Mobilio RS',
       'Honda Mobilio S', 'Honda Mobilio V', 'Honda WR-V Edge',
       'Honda WR-V SV', 'Honda WR-V VX', 'Honda WR-V i-DTEC',
       'Honda WR-V i-VTEC']


for feature in model_features:
    if feature not in df1.columns:
        if feature == fuel_type or feature == suspension_type or feature == car_model:
            df1[feature] = 1
        else:
            df1[feature] = 0
df1 = df1[model_features]

if pred:
    print("-"*36)
    print("df1 is\n", df1)
    print("shape of df1 is: ", df1.shape)
    print("-"*36)
    print("\n")
    if any([car_model is None, kms_driven is None, fuel_type is None, suspension_type is None]):
        st.error("Please, Select all Inputs before Pressing Predict Button.",icon="📝")
    else:
        prediction = model.predict(df1)[0]
        if prediction < 0:
            st.error("Predicted Price is Below Zero, Please select Valid Inputs. The year of car's model may be very low.", icon="⚠️")
        else:
            st.success(f"Predicted Price of Your Car is : ₹{prediction:,.0f}", icon="✅")