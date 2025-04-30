import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="milk_predict.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Kualitas Susu (Grade)")
st.write("Masukkan informasi susu untuk mendapatkan kulitas susu yang lebih baik.")

# Form input pengguna
pH = st.number_input("pH Susu", min_value=3.0, max_value=9.5)
Temperature = st.number_input("Temprature", min_value=34.0, max_value=90.0)
Taste = st.number_input("Taste", min_value=0.0, max_value=1.0)
Odor = st.number_input("Odor", min_value=0.0, max_value=1.0)
Fat = st.number_input("Fat)", min_value=0.0, max_value=1.0)
Turbidity = st.number_input("Turbidity", min_value=0.0, max_value=1.0)
Colour = st.number_input("Colour", min_value=244.0, max_value=255.0)

if st.button("Kualitas Susu"):
    # Preprocessing input
    input_data = np.array([[pH, Temperature, Taste, Odor, Fat, Turbidity, Colour]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    milk_name = label_encoder.inverse_transform([predicted_label])[0]


    st.success(f"Kualitas Susu: **{milk_name.upper()}**")