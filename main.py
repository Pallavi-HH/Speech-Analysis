import streamlit as st

emotion_process = st.Page("./emotion_detection/emotion_process.py", title="Process", default=True)
emotion_model = st.Page("./emotion_detection/emotion_model.py", title="Emotion Model")

depression_process = st.Page("./depression_detection/depression_process.py", title="Process")
phq9 = st.Page("./depression_detection/phq9.py", title="PHQ-9")
depression_model = st.Page("./depression_detection/depression_model.py", title="Depression Model")

pg = st.navigation(
        {
            "Emotion Detection": [emotion_process, emotion_model],
            "Depression Detection": [depression_process, phq9, depression_model]
        }
    )

pg.run()