# Diabetes Risk Prediction

A preliminary screening tool that estimates diabetes risk from health and lifestyle indicators. **Not a medical diagnosis.**

## Structure
```
streamlit_app.py   # Main interface (Streamlit)
app.py             # Alternative interface (Flask)
templates/         # Flask template
ml/                # train.ipynb, model.pkl, columns.pkl
requirements.txt
```

## Run
Requires Python 3.11+.
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens at http://localhost:8501. Flask alternative: `python app.py`.

> Install the exact scikit-learn version in `requirements.txt`; the model was trained with it.

## Model
- **Data:** CDC BRFSS 2015, 253,680 records, 21 features (~14% positive).
- **Compared:** Logistic Regression, Random Forest, Hist Gradient Boosting, XGBoost.
- **Final:** Hist Gradient Boosting, ROC-AUC ~0.827, threshold 0.4 (recall ~0.87).
- **Retraining:** place the Kaggle file `diabetes_binary_health_indicators_BRFSS2015.csv` in `ml/` and run `ml/train.ipynb`.

## Limitations
2015 US survey data, self-reported answers, low precision (~0.27). For screening only.