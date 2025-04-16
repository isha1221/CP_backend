import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load cleaned data
df = pd.read_csv("cleaned_career_data.csv")

# Features and labels
X = df["text"]
y = df["Role"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Build pipeline: TF-IDF + Classifier
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X, y_encoded)


joblib.dump(model, "app/ml/career_model.pkl")
joblib.dump(label_encoder, "app/ml/label_encoder.pkl")

print("Model trained and saved!")
