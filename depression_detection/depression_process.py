import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import seaborn as sns

st.title("🔬 Depression Detection Training Process")

st.markdown("""
This app explains the **CNN-based audio processing pipeline** for detecting depression.
We used **spectrogram images** from audio data to train a Convolutional Neural Network (CNN).
""")

# Section 1: Data Collection
st.header("📌 Step 1: Data Collection")
st.write("""
We used the **DAIC-WOZ dataset**, which contains:
- **Audio recordings** of 189 clinical interviews.
- **PHQ-8 depression scores** for labeling.
""")

# Section 2: Audio Processing
st.header("🎙️ Step 2: Audio Processing")
st.write("""
Speech is converted into **spectrogram images** to be used as CNN input.
Steps:
1. **Segmenting speech** to extract meaningful parts.
2. **Feature extraction** using Mel-frequency cepstral coefficients (MFCCs).
3. **Convert audio to spectrogram images**.
""")

def plot_spectrogram():
    y, sr = librosa.load(librosa.ex('trumpet'))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(6,3))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title("Spectrogram Example")
    st.pyplot(fig)

plot_spectrogram()

# Section 3: CNN Model Architecture
st.header("🧠 Step 3: CNN Model Architecture")
st.write("""
We trained a **Convolutional Neural Network (CNN)** to classify depression based on spectrogram images.
**Architecture:**
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 508, 508, 32)      2432      
                                                                 
 max_pooling2d (MaxPooling2  (None, 127, 127, 32)      0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 125, 125, 32)      9248      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 125, 41, 32)       0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 164000)            0         
                                                                 
 dense (Dense)               (None, 128)               20992128  
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 256)               33024     
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 1)                 257       
...
Total params: 21037089 (80.25 MB)
Trainable params: 21037089 (80.25 MB)
Non-trainable params: 0 (0.00 Byte)
```
""")


# Section 4: Model Evaluation
st.header("📊 Step 4: Model Evaluation")
st.write("""
Validation Results:
- **Accuracy:** 83.33%
- **Confusion Matrix:** The model was optimized using **Early Stopping, Dropout, and Regularization**.
""")

# Plot Confusion Matrix
def plot_confusion_matrix():
    cm = np.array([[43, 5], [11, 37]])
    labels = ["Non-Depressed", "Depressed"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

plot_confusion_matrix()

