import joblib

model = joblib.load("app/ml/career_model.pkl")
label_encoder = joblib.load("app/ml/label_encoder.pkl")

##one career prediction
# def predict_career_from_skills(skills_list: list[str]):
#     if not skills_list:
#         return ["No skills found to predict career"]

#     skill_text = ", ".join(skills_list).lower()

#     pred_encoded = model.predict([skill_text])[0]
#     pred_label = label_encoder.inverse_transform([pred_encoded])[0]

#     return [pred_label]


##3 career prediction with probabilities of matching jobs
def predict_top_careers(skills_list: list[str]):
    if not skills_list:
        return {"top_predictions": [], "probabilities": []}

    skill_text = ", ".join(skills_list).lower()

    # Get class probabilities
    probs = model.predict_proba([skill_text])[0]

    # Top 3 indices
    top_indices = probs.argsort()[-3:][::-1]

    top_careers = label_encoder.inverse_transform(top_indices)
    top_probs = [round(probs[i] * 100, 2) for i in top_indices]

    return {
        "top_predictions": list(top_careers),
        "probabilities": top_probs
    }