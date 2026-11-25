import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page configuration
st.set_page_config(page_title="Cancer Classifier", page_icon="🔬", layout="centered")

st.title("🔬 Skin / Tissue Cancer Classification")
st.write("Upload an image to convert it to grayscale and classify it as **Benign** or **Malignant**.")

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_trained_model():
    # 1. Rebuild VGG16
    conv_base = tf.keras.applications.VGG16(
        weights=None,
        include_top=False,
        input_shape=(150, 150, 3)
    )

    # 2. Match the EXACT trainable/frozen layers from your fine-tuning cell!
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == "block5_conv1":
            set_trainable = True
        layer.trainable = set_trainable

    # 3. Rebuild the Sequential model
    model = tf.keras.models.Sequential([
        conv_base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # 4. Build and load weights
    model.build((None, 150, 150, 3))
    model.load_weights("cancer_classifier.h5")
    
    return model


model = load_trained_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Open image and convert to Grayscale
    original_image = Image.open(uploaded_file)
    gray_image = original_image.convert("L")  # Converts to 1-channel grayscale

    # Display original and grayscale side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(gray_image, caption="Grayscale Input", use_container_width=True)

    # 2. Preprocess for VGG16 (Resize to 150x150, convert back to 3 channels, rescale)
    resized_gray = gray_image.resize((150, 150))
    rgb_gray = resized_gray.convert("RGB")  # Duplicates grayscale across 3 channels: (150, 150, 3)
    
    img_array = np.array(rgb_gray, dtype=np.float32) / 255.0  # Matches rescale=1./255
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 150, 150, 3)

    # 3. Predict
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Running prediction..."):
            prediction = model.predict(img_batch)[0][0]

        # Note: Keras flow_from_directory assigns classes alphabetically:
        # 0 = Benign, 1 = Malignant
        st.subheader("Prediction Result")
        if prediction >= 0.5:
            confidence = prediction * 100
            st.error(f"**Prediction: Malignant** (Confidence: {confidence:.2f}%)")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"**Prediction: Benign** (Confidence: {confidence:.2f}%)")

        st.caption("Disclaimer: This tool is for educational/research purposes only and is not a substitute for professional medical diagnosis.")
