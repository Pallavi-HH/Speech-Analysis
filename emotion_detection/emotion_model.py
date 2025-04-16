import librosa
import numpy as np
import python_speech_features
import streamlit as st
import torch
import os
from dotenv import load_dotenv
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
from st_audiorec import st_audiorec
from pyannote.audio.pipelines import SpeakerDiarization
from huggingface_hub import login
import tempfile

# Authenticate with Hugging Face (replace with your token)
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

login(HUGGING_FACE_TOKEN)

# Load the speaker diarization pipeline
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGING_FACE_TOKEN)

# Ensure FFmpeg is installed
AudioSegment.converter = "ffmpeg"

# Emotion categories
emotions = ["Anxiety", "Disgust", "Happy", "Boredom", "Anger", "Sadness", "Neutral"]

# Define Neural Network
class Net(nn.Module):
    def __init__(self, nb_features=27):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(nb_features, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 7)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
option = st.radio("Choose Input Method:", ["Upload Audio", "Record Audio"])

uploaded_file = None

temp_audio_path = None
if option == "Upload Audio":
    uploaded_file = st.file_uploader("📂 Upload an audio file (WAV format)", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name
elif option == "Record Audio":
    st.write("🎤 Click below to record your voice:")
    recorded_audio = st_audiorec()
    if recorded_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(recorded_audio)
            temp_audio_path = temp_audio.name

if temp_audio_path:
    st.audio(temp_audio_path, format="audio/wav")
    
    # Perform speaker diarization
    st.write("🔍 **Performing Speaker Diarization...**")
    diarization_result = pipeline({"uri": "audio", "audio": temp_audio_path})
    
    speaker_segments = {}
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((start, end))
    
    if "SPEAKER_00" in speaker_segments:
        speaker_audio = AudioSegment.silent(duration=0)
        audio = AudioSegment.from_wav(temp_audio_path)
        for start, end in speaker_segments["SPEAKER_00"]:
            segment = audio[int(start * 1000):int(end * 1000)]
            speaker_audio += segment
        
        extracted_audio_path = "./speaker_00.wav"
        speaker_audio.export(extracted_audio_path, format="wav")
        st.audio(extracted_audio_path, format="audio/wav")
        st.write("✅ **Extracted Speech of Speaker 00**")
        
        # Extract Features
        y, sr = librosa.load(extracted_audio_path, sr=16000)
        length = len(y)
        
        def extract_feature(f, y):
            feat = f(y=y, sr=16000)
            return np.mean(feat), np.std(feat), np.median(feat), np.max(feat), np.min(feat)
        
        mean_mfcc, sd_mfcc, median_mfcc, maxi_mfcc, mini_mfcc = extract_feature(librosa.feature.mfcc, y)
        mean_spectral_centroid, sd_spectral_centroid, median_spectral_centroid, maxi_spectral_centroid, mini_spectral_centroid = extract_feature(librosa.feature.spectral_centroid, y)
        mean_spectral_rolloff, sd_spectral_rolloff, median_spectral_rolloff, maxi_spectral_rolloff, mini_spectral_rolloff = extract_feature(librosa.feature.spectral_rolloff, y)
        
        feature_logfbank = python_speech_features.logfbank(y, 16000)
        mean_logfbank, sd_logfbank, median_logfbank, maxi_logfbank, mini_logfbank = np.mean(feature_logfbank), np.std(feature_logfbank), np.median(feature_logfbank), np.max(feature_logfbank), np.min(feature_logfbank)
        feature_spectral_subband_centroid = python_speech_features.ssc(y, 16000)
        mean_spectral_subband_centroid, sd_spectral_subband_centroid, median_spectral_subband_centroid, maxi_spectral_subband_centroid, mini_spectral_subband_centroid = \
            np.mean(feature_spectral_subband_centroid), np.std(feature_spectral_subband_centroid), np.median(feature_spectral_subband_centroid), np.max(feature_spectral_subband_centroid), np.min(feature_spectral_subband_centroid)
        
        ratio = np.sum(np.abs(y) < 0.01) / len(y)
        
        X = torch.tensor([
        length, mean_mfcc, sd_mfcc, median_mfcc, maxi_mfcc, mini_mfcc,
        mean_spectral_centroid, sd_spectral_centroid, median_spectral_centroid, maxi_spectral_centroid, mini_spectral_centroid,
        mean_spectral_rolloff, sd_spectral_rolloff, median_spectral_rolloff, maxi_spectral_rolloff, mini_spectral_rolloff,
        mean_logfbank, sd_logfbank, median_logfbank, maxi_logfbank, mini_logfbank,
        mean_spectral_subband_centroid, sd_spectral_subband_centroid, median_spectral_subband_centroid, maxi_spectral_subband_centroid, mini_spectral_subband_centroid,
        ratio
    ]).type(torch.FloatTensor)
        
        mu,std = X.mean(0), X.std(0)
        X_normalized = X.sub(mu).div(std)
        
        # Load Model and Predict
        model = Net()
        model.load_state_dict(torch.load("./models/fully_connected_nn_emotion_model/cross_val.pt", map_location=torch.device('cpu')))
        model.eval()
        
        output = model(X_normalized)
        probabilities = torch.softmax(output, dim=0).detach().numpy()
        
        max_index = np.argmax(probabilities)
        max_emotion = emotions[max_index]

        st.write(f"### Detected Emotion: {max_emotion}")
