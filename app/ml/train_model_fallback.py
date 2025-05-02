import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load cleaned data
df = pd.read_csv("categorized_jobs.csv")

# Drop any missing or bad rows
df = df.dropna(subset=["Key Skills", "Job Title"])
df = df[df["Key Skills"].astype(str).str.strip() != ""]
df = df[df["Job Title"].astype(str).str.strip() != ""]

# Optional: downsample for memory
df = df.sample(n=10000, random_state=42)

X = df["Key Skills"]
y = df["Job Title"]

# Encode job titles
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Build model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000,
     stop_words="english", ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train and save
model.fit(X, y_encoded)
joblib.dump(model, "app/ml/career_model.pkl")
joblib.dump(label_encoder, "app/ml/label_encoder.pkl")

print("✅ Model trained and saved successfully!")
