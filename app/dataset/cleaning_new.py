import pandas as pd
import re

# Load your CSV (from Kaggle)
df = pd.read_csv("app/dataset/jobs.csv")

df = df[["Job Title", "Key Skills"]]


def clean_job_title(title):
    if not isinstance(title, str):
        return ""
    title = re.sub(r'\s+', ' ', title.strip())
    title = re.sub(r'\s*\([^)]+\)\s*', '', title)  # Remove (stuff)
    title = re.sub(r'\s*-\s*.+$', '', title)  # Remove trailing after dashes
    return title.lower()

# Clean job skills


def clean_job_skills(skills):
    if not isinstance(skills, str):
        return ""
    skills = skills.lower().replace(',', ' ')
    skills = re.sub(r'[/\-]', ' ', skills)
    skills = re.sub(r'\s+', ' ', skills.strip())
    return skills


# Apply cleaners
df["Job Title"] = df["Job Title"].apply(clean_job_title)
df["Key Skills"] = df["Key Skills"].apply(clean_job_skills)

# Drop empty rows
df = df[(df["Job Title"] != "") & (df["Key Skills"] != "")]

# Save to clean file
df.to_csv("cleaned_job_data.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_job_data.csv'")
print(df.head().to_csv(index=False))
