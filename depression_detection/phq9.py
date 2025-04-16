import streamlit as st

st.title("PHQ-9 Questionnaire")

questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed, or the opposite—being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself"
]

options = ["Not at all", "Several days", "More than half the days", "Nearly every day"]

responses = []
for i, question in enumerate(questions):
    response = st.radio(f"{i+1}. {question}", options, index=0, key=f"q{i}")
    responses.append(options.index(response))

if st.button("Submit"):
    total_score = sum(responses)
    st.write("## Your PHQ-9 Score: ", total_score)

    if total_score >= 20:
        st.error("Severe Depression: Consider seeking professional help immediately.")
    elif total_score >= 15:
        st.warning("Moderately Severe Depression: It may be beneficial to consult a healthcare provider.")
    elif total_score >= 10:
        st.info("Moderate Depression: Consider speaking with a healthcare provider if symptoms persist.")
    elif total_score >= 5:
        st.success("Mild Depression: Monitoring your mental health and self-care strategies may help.")
    else:
        st.success("Minimal or No Depression: Your score suggests no significant depressive symptoms.")

    st.write("\n**Disclaimer:** This is not a clinical diagnosis. If you're struggling, consider reaching out to a mental health professional.")
