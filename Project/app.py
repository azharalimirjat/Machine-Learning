# import streamlit as st
# import pandas as pd
# import joblib


# model = joblib.load("KNN_heart.pkl")
# model_scaler = joblib.load("Scaler_heart.pkl")
# model_columns = joblib.load("heart_columns.pkl")

# st.title("Heart Stroke Prediction")
# st.markdown("Provide the following details")

# age = st.slider("Age", 18,100,40)
# sex = st.selectbox("SEX",['M','F'])
# chestpain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
# restingBP = st.number_input("Restin Blood Pressure (mm Hg)", 80,200,120)
# cholestrol = st.number_input("Cholestrol (mg/dL)", 100,600,200)
# fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0,1])
# restingECG = st.selectbox("Resting ECG", ['Normal', 'ST', "LVH"])
# max_hr = st.slider("Max Heart Rate", 60,220,150)
# exercise_angina = st.selectbox("Exercise-Included Angina", ["Y","N"])
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0,6.0,1.0)
# st_slope = st.selectbox("ST Slope", ["Up","Flat", "Down"])


# if st.button("Predict"):
#     raw_input = {
#         'Age' : age,
#         'RestingBP' : restingBP,
#         'Cholestrol' : cholestrol,
#         'FastingBS' : fastingBS,
#         'MaxHR' : max_hr,
#         'Oldpeak' : oldpeak,
#         'Sex_' + sex: 1,
#         'ChestPainType_' + chestpain: 1,
#         'RestingECG_' + restingECG :1,
#         'ExerciseAngina_' + exercise_angina : 1,
#         'ST_Slope_' + st_slope : 1
#     }
    
#     input_df = pd.DataFrame([raw_input])
    
#     for col in model_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     input_df = input_df[model_columns]

#     scaled_input = model_scaler.transform(input_df)
    
#     prediction = model.predict(scaled_input)[0]

#     if prediction == 1:
#         st.error("High Risk of Heart Disease")
#     else:
#         st.success("Low Risk of Heart Disease")    
    
import streamlit as st
import pandas as pd
import joblib

# Load model components
model = joblib.load("KNN_heart.pkl")
model_scaler = joblib.load("Scaler_heart.pkl")  # ✅ This must be a StandardScaler or similar
model_columns = joblib.load("heart_columns.pkl")  # ✅ List of all expected columns

st.title("❤️ Heart Disease Risk Prediction")
st.markdown("Please provide the following details:")

# Input fields
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chestpain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholestrol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restingECG = st.selectbox("Resting ECG", ['Normal', 'ST', "LVH"])
max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # Create one-hot style input
    raw_input = {
        'Age': age,
        'RestingBP': restingBP,
        'Cholestrol': cholestrol,
        'FastingBS': fastingBS,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chestpain}': 1,
        f'RestingECG_{restingECG}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }

    # Create DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing one-hot columns as 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match model
    input_df = input_df[model_columns]

    try:
        # Scale input
        scaled_input = model_scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Show result
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
