import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# === Load the Model (.h5) ===
@st.cache_resource
def load_model():
    model_path = "C:\Users\madhu\Desktop\cauliflower project\app\models\densenet5050.h5"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
# dfghjkl ,adhu is good girlfghjk
# === Class Labels (from dataset folders) ===
class_labels = [
    "Alternaria_Leaf_Spot",
    "Bacterial spot rot",
    "Black Rot",
    "Cabbage aphid colony",
    "club root",
    "Downy Mildew",
    "No disease",
    "ring spot"
]

# === Streamlit UI ===
st.title("🥦 Cauliflower Disease Detector")
st.write("Upload one cauliflower leaf image to predict its disease.")

uploaded_file = st.file_uploader("📁 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    confidence = predictions[0][pred_index] * 100
    predicted_class = class_labels[pred_index]

    # Display result
    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
