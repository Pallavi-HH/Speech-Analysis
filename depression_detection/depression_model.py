import streamlit as st
import os
from dotenv import load_dotenv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pydub import AudioSegment
from pyannote.audio.pipelines import SpeakerDiarization
from huggingface_hub import login
import tempfile
from st_audiorec import st_audiorec

# Set Hugging Face API Token
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Authenticate with Hugging Face
login(HUGGING_FACE_TOKEN)

# Load Pyannote Speaker Diarization pipeline
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGING_FACE_TOKEN)

# Load trained CNN model
model = load_model("./models/cnn_depressed_model")

# Streamlit UI
st.title("🎙️ Depression Detection from Speech")
st.markdown("Upload or **record** an audio file, and this app will analyze it to determine if the speaker shows signs of **depression**.")

# Option to upload or record audio
option = st.radio("Choose Input Method:", ["Upload Audio", "Record Audio"])

uploaded_file = None
temp_audio_path = None

# Handling Audio Input
if option == "Upload Audio":
    uploaded_file = st.file_uploader("📂 Upload an audio file (WAV format)", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.read())  # Read file bytes
            temp_audio_path = temp_audio.name  # Store temp file path

elif option == "Record Audio":
    st.write("🎤 Click below to record your voice:")
    recorded_audio = st_audiorec()
    if recorded_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(recorded_audio)  # Save recorded bytes
            temp_audio_path = temp_audio.name

# Ensure we have a valid audio file before proceeding
if temp_audio_path:
    st.audio(temp_audio_path, format="audio/wav", start_time=0)

    # Step 1: Speaker Diarization
    st.write("🔍 Performing **Speaker Diarization**...")

    def diarize_audio(audio_file):
        """Performs speaker diarization and returns speaker segments"""
        diarization_result = pipeline({"uri": "audio", "audio": audio_file})

        speaker_segments = {}
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            start, end = turn.start, turn.end
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((start, end))

        return speaker_segments

    speaker_timestamps = diarize_audio(temp_audio_path)

    # Step 2: Extract Speaker 00
    st.write("🎙️ Extracting **Speaker 00's Voice**...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as extracted_audio_file:
        output_audio_path = extracted_audio_file.name  # Store temp file path

    def segment_audio(input_audio, speaker_segments, output_file):
        """Extracts the first speaker's segments and saves them as an audio file"""
        audio = AudioSegment.from_wav(input_audio)

        if "SPEAKER_00" in speaker_segments.keys():
            speaker_audio = AudioSegment.silent(duration=0)
            for start, end in speaker_segments["SPEAKER_00"]:
                segment = audio[int(start * 1000):int(end * 1000)]  # Convert sec → ms
                speaker_audio += segment
            speaker_audio.export(output_file, format="wav")
            return output_file
        else:
            return None

    extracted_audio = segment_audio(temp_audio_path, speaker_timestamps, output_audio_path)

    if extracted_audio:
        st.audio(output_audio_path, format="audio/wav")
        st.write("✅ **Extracted Speech of Speaker 00**")

        # Step 3: Generate Spectrogram
        st.write("🎨 Generating **Spectrogram**...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as spectrogram_file:
            spectrogram_path = spectrogram_file.name  # Store temp file path

        def create_spectrogram(audio_path, output_image):
            """Creates and saves a spectrogram from an audio file"""
            y, sr = librosa.load(audio_path, sr=16000)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

            fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            librosa.display.specshow(D, sr=sr, cmap='inferno', ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            plt.savefig(output_image, dpi=100, pad_inches=0)
            plt.close(fig)

        create_spectrogram(output_audio_path, spectrogram_path)

        st.image(spectrogram_path, caption="Generated Spectrogram", use_container_width=True)

        # Step 4: Make Prediction
        st.write("🧠 Running **Depression Prediction**...")

        def preprocess_image(image_path, target_size=(512, 512)):
            """Prepares spectrogram image for CNN model"""
            img = image.load_img(image_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array

        processed_img = preprocess_image(spectrogram_path)
        prediction = model.predict(processed_img)[0][0]

        # Step 5: Display Prediction Result
        if prediction > 0.5:
            st.error("🔴 **Prediction: Depressed**")
        else:
            st.success("🟢 **Prediction: Not Depressed**")

    else:
        st.warning("⚠️ No valid speech segments found for Speaker 00!")

else:
    st.warning("⚠️ Please upload or record an audio file!")
