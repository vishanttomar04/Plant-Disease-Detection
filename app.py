import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Plant Disease Detector",
    page_icon="🌱"
)

st.title("🌱 AI Plant Disease Detector")
st.write("Upload a plant leaf image and let the CNN model predict its condition.")

MODEL_PATH = "model/plant_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Healthy",
    "Potato - Late Blight",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Healthy",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Mosaic Virus",
    "Tomato - Yellow Leaf Curl Virus"
]

uploaded_file = st.file_uploader(
    "📷 Upload any supported plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf",
        use_container_width=True
    )

    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    if st.button("🔍 Detect Plant Disease"):

        predictions = model.predict(image_array, verbose=0)[0]

        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

        st.success(f"🌿 Prediction: {predicted_class}")
        st.info(f"🎯 Confidence: {confidence:.2f}%")

        st.warning(
            "This AI model is for educational purposes. "
            "Predictions may be inaccurate and should not replace "
            "professional agricultural advice."
        )