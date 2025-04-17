# import pandas as pd

# # Step 1: Load both CSV files
# skills_df = pd.read_csv('job_skills.csv')
# jobs_df = pd.read_csv('linkedin_job_postings.csv')

# # Step 2: Merge based on 'job_link'
# merged_df = pd.merge(jobs_df[['job_link', 'job_title']], skills_df[['job_link', 'job_skills']], on='job_link', how='inner')

# # Step 3: Save to a new CSV

# final_df = merged_df[['job_title', 'job_skills']]

# # Save to new CSV
# final_df.to_csv('job_title_with_skills.csv', index=False)

# print("✅ Merged CSV created: 'merged_job_title_skills.csv'")


# modified::

import pandas as pd
import re

# Load your CSV (from Kaggle)
df = pd.read_csv("app/dataset/jobs.csv")

# Only keep necessary columns (adjust as per your file)
df = df[["Job Title", "Key Skills"]]

# Clean job titles
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
