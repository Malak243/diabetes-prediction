import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, "ml", "model.pkl"))
columns = joblib.load(os.path.join(BASE, "ml", "columns.pkl"))
THRESHOLD = 0.4

YN = [(1, "نعم"), (0, "لا")]

def yn(name, label):
    return {"name": name, "label": label, "type": "select", "options": YN}

FIELDS = [
    {"name": "BMI", "label": "مؤشر كتلة الجسم BMI", "type": "number", "min": 10, "max": 100, "value": 25},
    {"name": "Age", "label": "الفئة العمرية", "type": "select", "options": [
        (1, "18-24"), (2, "25-29"), (3, "30-34"), (4, "35-39"), (5, "40-44"),
        (6, "45-49"), (7, "50-54"), (8, "55-59"), (9, "60-64"), (10, "65-69"),
        (11, "70-74"), (12, "75-79"), (13, "80+")]},
    {"name": "Sex", "label": "الجنس", "type": "select", "options": [(0, "أنثى"), (1, "ذكر")]},
    {"name": "GenHlth", "label": "تقييمك لصحتك العامة", "type": "select", "options": [
        (1, "ممتازة"), (2, "جيدة جداً"), (3, "جيدة"), (4, "مقبولة"), (5, "ضعيفة")]},
    {"name": "MentHlth", "label": "أيام سوء الصحة النفسية (آخر 30 يوم)", "type": "number", "min": 0, "max": 30, "value": 0},
    {"name": "PhysHlth", "label": "أيام سوء الصحة الجسدية (آخر 30 يوم)", "type": "number", "min": 0, "max": 30, "value": 0},
    {"name": "Education", "label": "المستوى التعليمي", "type": "select", "options": [
        (1, "لا تعليم"), (2, "ابتدائي"), (3, "إعدادي/ثانوي غير مكتمل"),
        (4, "ثانوي"), (5, "جامعي غير مكتمل"), (6, "خريج جامعة")]},
    {"name": "Income", "label": "مستوى الدخل (1 الأقل ← 8 الأعلى)", "type": "select",
     "options": [(i, str(i)) for i in range(1, 9)]},
    yn("HighBP", "ضغط دم مرتفع؟"),
    yn("HighChol", "كوليسترول مرتفع؟"),
    yn("CholCheck", "فحصتِ الكوليسترول في آخر 5 سنين؟"),
    yn("Smoker", "دخّنتِ 100 سيجارة على الأقل في حياتك؟"),
    yn("Stroke", "عندك تاريخ سكتة دماغية؟"),
    yn("HeartDiseaseorAttack", "أمراض قلب أو جلطة؟"),
    yn("PhysActivity", "نشاط بدني في آخر 30 يوم؟"),
    yn("Fruits", "بتاكلي فاكهة يومياً؟"),
    yn("Veggies", "بتاكلي خضار يومياً؟"),
    yn("HvyAlcoholConsump", "استهلاك كحول عالٍ؟"),
    yn("AnyHealthcare", "عندك تأمين صحي؟"),
    yn("NoDocbcCost", "تجنبتِ الطبيب بسبب التكلفة؟"),
    yn("DiffWalk", "صعوبة في المشي أو صعود السلم؟"),
]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    values = {}
    if request.method == "POST":
        values = request.form.to_dict()
        row = {c: float(values[c]) for c in columns}
        X = pd.DataFrame([row], columns=columns)
        proba = float(model.predict_proba(X)[0, 1])
        result = {"percent": round(proba * 100, 1), "high": proba >= THRESHOLD}
    return render_template("index.html", fields=FIELDS, result=result, values=values)

if __name__ == "__main__":
    app.run(debug=True)