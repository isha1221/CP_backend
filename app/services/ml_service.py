import joblib

model = joblib.load("app/ml/career_model.pkl")
label_encoder = joblib.load("app/ml/label_encoder.pkl")


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
