import pandas as pd

df = pd.read_csv("./app/dataset/role_based_on_skills_test.csv")

df = df[["Role", "text"]]


df = df.dropna(subset=["Role", "text"])

df["Role"] = df["Role"].str.lower()
df["text"] = df["text"].str.strip().str.title()

# df = df.drop_duplicates()


df.to_csv("cleaned_career_data.csv", index=False)