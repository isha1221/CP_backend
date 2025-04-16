import joblib

model = joblib.load("app/ml/career_model.pkl")
label_encoder = joblib.load("app/ml/label_encoder.pkl")

def predict_career_from_skills(skills_list: list[str]):
    if not skills_list:
        return ["No skills found to predict career"]

    skill_text = ", ".join(skills_list).lower()

    pred_encoded = model.predict([skill_text])[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return [pred_label]
