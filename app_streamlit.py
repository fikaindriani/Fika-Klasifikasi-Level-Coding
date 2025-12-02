import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Klasifikasi Level Coding",
    page_icon="🔥"
)

model = joblib.load("klasifikasi_level_coding.joblib")

st.title("🔥Klasifikasi Level Coding🔥")
st.markdown("Klasifikasi level coding berdasarkan fitur hours coding daily, preferred language, typing speed, import usage, dan OOP usage")

# Input
hours_coding_daily = st.slider("hours_coding_daily", 2.5, 5.5, 3.0)
preferred_language = st.selectbox("preferred_language", ["Python", "C++", "Java"])
typing_speed = st.slider("typing_speed", 25, 65, 35)
import_usage = st.pills("import_usage", ["Yes", "No"])
oop_usage = st.radio("oop_usage", ["Yes", "No"])

# Prediksi
if st.button("prediksi", type="primary"):
    data_baru = pd.DataFrame([[hours_coding_daily, preferred_language, typing_speed, import_usage, oop_usage]],
                             columns=["hours_coding_daily", "preferred_language", "typing_speed", "import_usage", "oop_usage"])

    prediksi = model.predict(data_baru)[0]
    presentase = max(model.predict_proba(data_baru)[0])

    st.success(f"Model Memprediksi {prediksi} dengan tingkat keyakinan {presentase*100:.2f}%")
    st.balloons()

st.divider()
st.caption("Model Ini Di Buat Oleh **Fika💜💟**")

