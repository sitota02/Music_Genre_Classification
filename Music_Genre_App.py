import os
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import get_custom_objects

# Register custom layers (if you have any custom layers, replace this with actual layers you define)
if "MyCustomLayer" not in get_custom_objects():  # Check if already registered
    class MyCustomLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs  # replace with custom behavior

# Function to load the trained model with custom layers registered
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Trained_model.keras", custom_objects={"MyCustomLayer": MyCustomLayer})

# Function to load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    if len(audio_data) < chunk_samples:
        raise ValueError("Audio file is too short for processing. Please upload a longer file.")

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    data = []

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = tf.image.resize(
            tf.convert_to_tensor(np.expand_dims(mel_spectrogram, axis=-1), dtype=tf.float32), 
            target_shape
        ).numpy()
        data.append(mel_spectrogram)
    
    return np.array(data)

# Create a simple CNN model
def create_model(input_shape=(150, 150, 1)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes for 10 genres
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(X_train, y_train):
    model = create_model(input_shape=(150, 150, 1))
    model.fit(X_train, y_train, epochs=5)
    model.save("Trained_model.keras")
    return model

# Function to predict the genre using the model
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Streamlit Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Train Model", "Prediction"])

# Home Page
if app_mode == "Home":
    st.markdown("""<style> .stApp { background-color: #181646; color: white; } h2, h3 { color: white; } </style>""", unsafe_allow_html=True)
    st.markdown(""" ## Welcome to the Music Genre Classification System! 🎶🎧 **Upload an audio file to detect its genre!**  """)
    
    # Ensure the image exists in the correct directory or provide a placeholder
    image_path = "chachalu.png"
    if os.path.exists(image_path):
        st.image(image_path)  # Removed 'use_container_width=True'
    else:
        st.warning("Image not found, using default!")
        st.image("https://via.placeholder.com/150")

# About Page
elif app_mode == "About Project":
    st.markdown(""" ### About the Project The Music Genre Classification System uses AI to analyze audio files and predict their genres. It processes the audio into Mel spectrograms and classifies them using a trained deep learning model.
     """)

# Train Model Page
elif app_mode == "Train Model":
    st.header("Train Model")
    uploaded_file = st.file_uploader("Upload Training Data", type=["mp3"])

    if uploaded_file:
        # Save the uploaded file
        upload_dir = "Train_Music"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        filepath = os.path.join(upload_dir, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

        # Preprocess the uploaded audio and generate dummy labels for training
        X_train = load_and_preprocess_data(filepath)
        y_train = np.random.randint(0, 10, size=X_train.shape[0])  # Dummy labels (10 genres)

        if st.button("Train and Save Model"):
            model = train_model(X_train, y_train)
            st.success("Model Trained and Saved!")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])

    if test_mp3:
        # Ensure the directory exists before saving the file
        upload_dir = "Test_Music"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        filepath = os.path.join(upload_dir, test_mp3.name)
        with open(filepath, "wb") as f:
            f.write(test_mp3.read())
        
        if st.button("Play Audio"):
            st.audio(filepath)

        if st.button("Predict"):
            with st.spinner("Processing..."):
                try:
                    # Preprocess the uploaded audio
                    X_test = load_and_preprocess_data(filepath)
                    # Get model prediction
                    result_index = model_prediction(X_test)
                    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                    st.balloons()
                    st.success(f"Model Prediction: It's a {genres[result_index]} music!")
                except Exception as e:
                    st.error(f"Error: {e}")
