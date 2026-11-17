import os, joblib, pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "ml", "model.pkl"))
columns = joblib.load(os.path.join(BASE, "ml", "columns.pkl"))
THRESHOLD = 0.4

st.set_page_config(page_title="Diabetes Risk", layout="centered")
st.title("التنبؤ بخطر الإصابة بالسكري")
st.caption("أداة فحص مبدئي، مش تشخيص طبي.")

def yn(label):
    return 1 if st.selectbox(label, ["لا", "نعم"]) == "نعم" else 0

c1, c2 = st.columns(2)
with c1:
    bmi = st.number_input("BMI", 10.0, 100.0, 25.0)
    age = st.selectbox("الفئة العمرية (1=18-24 ... 13=80+)", list(range(1, 14)))
    sex = st.selectbox("الجنس", ["أنثى", "ذكر"]) == "ذكر"
    genhlth = st.selectbox("الصحة العامة (1 ممتازة ... 5 ضعيفة)", [1, 2, 3, 4, 5])
    ment = st.slider("أيام سوء الصحة النفسية", 0, 30, 0)
    phys = st.slider("أيام سوء الصحة الجسدية", 0, 30, 0)
    edu = st.selectbox("التعليم (1-6)", list(range(1, 7)), index=3)
    income = st.selectbox("الدخل (1-8)", list(range(1, 9)), index=4)
with c2:
    highbp = yn("ضغط مرتفع؟")
    highchol = yn("كوليسترول مرتفع؟")
    cholcheck = yn("فحصتِ الكوليسترول آخر 5 سنين؟")
    smoker = yn("مدخّنة؟")
    stroke = yn("سكتة دماغية؟")
    heart = yn("أمراض قلب؟")
    physact = yn("نشاط بدني؟")
    fruits = yn("فاكهة يومياً؟")
    veggies = yn("خضار يومياً؟")
    alcohol = yn("كحول عالٍ؟")
    anyhc = yn("تأمين صحي؟")
    nodoc = yn("تجنبتِ الطبيب للتكلفة؟")
    diffwalk = yn("صعوبة في المشي؟")

if st.button("احسب الخطورة", type="primary"):
    vals = {"HighBP": highbp, "HighChol": highchol, "CholCheck": cholcheck,
            "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
            "HeartDiseaseorAttack": heart, "PhysActivity": physact,
            "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": alcohol,
            "AnyHealthcare": anyhc, "NoDocbcCost": nodoc, "GenHlth": genhlth,
            "MentHlth": ment, "PhysHlth": phys, "DiffWalk": diffwalk,
            "Sex": int(sex), "Age": age, "Education": edu, "Income": income}
    X = pd.DataFrame([vals], columns=columns)
    p = float(model.predict_proba(X)[0, 1])
    st.progress(min(p, 1.0))
    if p >= THRESHOLD:
        st.error(f"نسبة الخطورة: {p*100:.1f}% ، مرتفعة نسبياً. يُنصح بتحليل سكر واستشارة طبيب.")
    else:
        st.success(f"نسبة الخطورة: {p*100:.1f}% ، منخفضة نسبياً.")