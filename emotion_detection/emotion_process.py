import streamlit as st

# Title
st.title("Emotion Detection Training Process")

# Introduction
st.write("""
This app explains the emotion detection training process using the **Berlin Database of Emotional Speech**.
We use **fully connected neural networks** (FCNN) with extracted audio features.
""")

# Step 1: Dataset Used
st.header("1. Berlin Database of Emotional Speech")
st.write("""
The dataset consists of emotional speech recordings labeled with emotions such as Anxiety, Disgust, Happy, Boredom, Anger, Sadness, and Neutral.
""")

# Step 2: Feature Extraction
st.header("2. Feature Extraction")
st.write("""
The following features were extracted from the audio:
- **Length of Audio**
- **MFCC Features**: Mean, Standard Deviation, Median, Max, Min
- **Spectral Centroid**: Mean, SD, Median, Max, Min
- **Spectral Rolloff**: Mean, SD, Median, Max, Min
- **Log Filter Bank Features**: Mean, SD, Median, Max, Min
- **Spectral Subband Centroid**: Mean, SD, Median, Max, Min
- **Silence-to-Speech Ratio**
""")

# Step 3: Model Architecture
st.header("3. Fully Connected Neural Network (FCNN)")
st.write("""
The model consists of 4 fully connected layers:

- **Input Layer**: 27 Features
- **Hidden Layers**: 3 layers with 200 neurons each
- **Output Layer**: 7 classes representing different emotions

The activation function used is ReLU, and the model is trained using Cross Entropy Loss.
""")

# Neural Network Model Visualization
st.subheader("Model Architecture")
st.code('''
Net(
  (fc1): Linear(in_features=27, out_features=200, bias=True)
  (fc2): Linear(in_features=200, out_features=200, bias=True)
  (fc3): Linear(in_features=200, out_features=200, bias=True)
  (fc4): Linear(in_features=200, out_features=7, bias=True)
)
''', language='python')

# Step 4: Training and Accuracy
st.header("4. Model Training and Evaluation")
st.write("""
The model was trained and evaluated on the dataset, achieving the following accuracy:

| Emotion   | Accuracy (%) |
|-----------|-------------|
| Global    | 66.17       |
| Anxiety   | 62.76       |
| Disgust   | 61.07       |
| Happy     | 56.77       |
| Boredom   | 62.64       |
| Anger     | 79.24       |
| Sadness   | 84.6        |
| Neutral   | 63.22       |
""")

# Step 5: Confusion Matrix
st.header("5. Confusion Matrix")
st.image("./emotion_detection/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
