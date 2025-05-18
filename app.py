from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('osteoarthritis_comorbidity_model.pkl')
label_mapping = joblib.load('label_mapping.pkl')

# Clinical notes function as per your logic
def clinical_notes(bmi_class_asian, pain_score, oa_severity, smoking):
    notes = []
    if bmi_class_asian in ["overweight", "obese"] and oa_severity >= 3:
        notes.append("Recommend weight management for OA relief")
    if pain_score > 7:
        notes.append("High pain, consider advanced imaging")
    if smoking == 1:
        notes.append("Advise smoking cessation")
    return notes

# BMI classification (Asian) helper, to get bmi_class_asian for clinical notes
def classify_bmi_asian(bmi):
    if bmi < 17.5:
        return "underweight"
    elif 17.5 <= bmi <= 22.99:
        return "normal"
    elif 23 <= bmi <= 27.99:
        return "overweight"
    else:
        return "obese"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        age = float(data['age'])
        bmi = float(data['bmi'])
        oa_severity = int(data['oa_severity'])
        activity = int(data['activity'])
        smoking = int(data['smoking'])
        pain_score = float(data['pain_score'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

    features = np.array([[age, bmi, oa_severity, activity, smoking, pain_score]])
    pred_code = model.predict(features)[0]
    pred_label = label_mapping.get(pred_code, "Unknown")

    bmi_class_asian = classify_bmi_asian(bmi)
    notes = clinical_notes(bmi_class_asian, pain_score, oa_severity, smoking)

    return jsonify({
        'comorbidity': pred_label,
        'clinical_notes': notes
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
