import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pickle

# Title
st.title("Student Prediction and Clustering App")

# Data input
st.header("Data Input")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=[0], error_bad_lines=False, sep=";")
    
    # Drop specified columns
    column_drop = ['tgl_lahir', 'current_date']
    df.drop(columns=column_drop, inplace=True)
    
    # Label encode 'class' column
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    
    # Remove non-numeric characters from 'rata-nilai' column and convert to float
    df['rata-nilai'] = df['rata-nilai'].str.replace(',', '.').astype(float)
    
    # Convert float values to integers
    df['rata-nilai'] = df['rata-nilai'].astype(int)

    # Display basic information about the modified DataFrame
    st.write("Modified DataFrame Info:")
    st.write(df.info())

    # Data cleaning and processing
    # You can include your data preprocessing steps here

    # Student Prediction Section
    st.header("Student Prediction")

    # Dropdown for student selection
    student_selection = st.selectbox("Select a Student", df.index)

    # Load the trained RandomForestRegressor model
    with open('random_forest_model.pkl', 'rb') as model_file:
        random_forest_model = pickle.load(model_file)

    # Get the selected student's data
    selected_student_data = df.loc[student_selection]

    # Make predictions using the RandomForestRegressor model
    prediction = random_forest_model.predict([selected_student_data])[0]

    # Display the prediction score
    st.subheader("Prediction Score")
    st.write(f"The predicted score for the selected student is: {prediction:.2f}")

    # Predictions of first to six semester
    # Include your prediction logic here and display the results

    # Student Cluster Section
    st.header("Student Cluster")

    # Interactive cluster plot
    # Include your code for plotting the interactive cluster plot here

    # Display the Streamlit app
    st.pyplot()  # This will display any plots you create
