import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import google.generativeai as genai
import os
import numpy as np

# Load the dataset
df = pd.read_excel("heart.xlsx", sheet_name="Heart2")

# Clean the data - handle missing values if any
df = df.dropna()

# Domain-Specific Augmentation
def medical_augmentation(df):
    augmented = []
    
    for _, row in df.iterrows():
        # Create variations based on clinical relationships
        for _ in range(2):  # Create 2 augmented samples per original
            new_row = row.copy()
            
            # If patient has high cholesterol, likely higher blood pressure
            if new_row['Chol'] > 240:
                new_row['RestBP'] += np.random.randint(5, 15)
                
            # If patient has exercise induced angina, likely higher ST depression
            if new_row['ExAng'] == 1:
                new_row['Oldpeak'] += np.random.uniform(0.1, 0.5)
                
            augmented.append(new_row)
    
    return pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)

df = medical_augmentation(df)
df = df.drop_duplicates()

from sklearn.preprocessing import LabelEncoder

# At the beginning of your script (after loading data)
all_chest_pain_types = ['typical', 'atypical', 'non-anginal', 'asymptomatic']
all_thal_types = ['normal', 'fixed defect', 'reversible defect']

# Initialize LabelEncoders with all possible categories
chest_pain_encoder = LabelEncoder().fit(all_chest_pain_types)
thal_encoder = LabelEncoder().fit(all_thal_types)
le = LabelEncoder()
le.fit(all_chest_pain_types)  # Fit with all possible categories
df['AHD'] = le.fit_transform(df['AHD'])
df['ChestPain'] = le.fit_transform(df['ChestPain'])
df['Thal'] = le.fit_transform(df['Thal'].astype(str))  # Handle NA values

# Select features and target
features = ['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'Fbs', 'RestECG', 
            'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca', 'Thal']
X = df[features]
y = df['AHD']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
ml_model = RandomForestClassifier(random_state=42)
ml_model.fit(X_train, y_train)

# Evaluate
y_pred = ml_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(ml_model, 'heart_disease_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Configure Gemini
genai.configure(api_key='AIzaSyD-pyrqtBxbipwtRHMvghCRxOQi49O_3ps')

# Initialize the model
model_name = 'gemini-2.0-flash'
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}


gemini_model = genai.GenerativeModel(model_name=model_name,
                            generation_config=generation_config)


import streamlit as st

# Load the saved model and encoder
model = joblib.load('heart_disease_model.pkl')
le = joblib.load('label_encoder.pkl')

def predict_ahd(input_data):
    """Predict AHD based on input features"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Convert categorical variables
        input_df['ChestPain'] = chest_pain_encoder.transform([input_data['ChestPain']])[0]
        input_df['Thal'] = thal_encoder.transform([str(input_data['Thal'])])[0]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return prediction, probability
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

def generate_explanation(input_data, prediction, probability):
    """Generate natural language explanation using Gemini"""
    prompt = f"""
    A patient with the following characteristics:
    - Age: {input_data['Age']}
    - Sex: {'Male' if input_data['Sex'] == 1 else 'Female'}
    - Chest Pain Type: {input_data['ChestPain']}
    - Resting Blood Pressure: {input_data['RestBP']} mmHg
    - Cholesterol: {input_data['Chol']} mg/dl
    - Fasting Blood Sugar > 120 mg/dl: {'Yes' if input_data['Fbs'] == 1 else 'No'}
    - Resting ECG Results: {input_data['RestECG']}
    - Maximum Heart Rate Achieved: {input_data['MaxHR']}
    - Exercise Induced Angina: {'Yes' if input_data['ExAng'] == 1 else 'No'}
    - ST Depression Induced by Exercise: {input_data['Oldpeak']}
    - Slope of Peak Exercise ST Segment: {input_data['Slope']}
    - Number of Major Vessels Colored by Fluoroscopy: {input_data['Ca']}
    - Thalassemia: {input_data['Thal']}
    
    Has a {'high' if probability > 0.7 else 'moderate' if probability > 0.5 else 'low'} probability ({probability*100:.1f}%) of having angiographic heart disease (AHD).
    
    Please provide a detailed explanation in simple terms for a non-medical person about what this prediction means, which factors contributed most to this prediction, and what they should do next.
    """
    
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Heart Disease Prediction Chatbot")

st.write("""
Please enter the patient's information to predict the likelihood of angiographic heart disease (AHD).
""")

# Input form
with st.form("patient_info"):
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["typical", "atypical", "non-anginal", "asymptomatic"])
    rest_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    rest_ecg = st.selectbox("Resting ECG Results", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    ex_ang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversible defect"])
    
    submitted = st.form_submit_button("Predict")
    
if submitted:
    # Prepare input data
    input_data = {
        'Age': age,
        'Sex': 1 if sex == "Male" else 0,
        'ChestPain': chest_pain,
        'RestBP': rest_bp,
        'Chol': chol,
        'Fbs': 1 if fbs == "Yes" else 0,
        'RestECG': ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"].index(rest_ecg),
        'MaxHR': max_hr,
        'ExAng': 1 if ex_ang == "Yes" else 0,
        'Oldpeak': oldpeak,
        'Slope': ["upsloping", "flat", "downsloping"].index(slope) + 1,
        'Ca': ca,
        'Thal': thal
    }
    
    # Make prediction
    prediction, probability = predict_ahd(input_data)
    
    if prediction is not None:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High risk of AHD (Probability: {probability*100:.1f}%)")
        else:
            st.success(f"Low risk of AHD (Probability: {probability*100:.1f}%)")
        
        # Generate explanation
        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(input_data, prediction, probability)
            st.subheader("Explanation")
            st.write(explanation)
    else:
        st.error("Error in making prediction. Please check your inputs.")                            