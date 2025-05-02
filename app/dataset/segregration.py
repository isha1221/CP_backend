import pandas as pd

# Define skill categories and associated keywords
skill_categories = {
    'Frontend Developer': ['html', 'css', 'javascript', 'jquery', 'react', 'angular', 'vue', 'front end', 'frontend', 'ui'],
    'Backend Developer': ['java', 'python', 'c#', '.net', 'php', 'nodejs', 'sql', 'database', 'api', 'server', 'backend', 'back end', 'weblogic', 'jms', 'soap'],
    'ML Developer': ['machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning', 'data science', 'neural networks', 'tensorflow', 'pytorch'],
    'Full Stack Developer': ['full stack', 'fullstack', 'frontend', 'backend', 'front end', 'back end'],
    'DevOps Engineer': ['aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'devops', 'git'],
    'Mobile Developer': ['android', 'ios', 'swift', 'kotlin', 'mobile', 'react native', 'flutter'],
    'Security Engineer': ['security', 'oim', 'oam', 'idam', 'oauth', 'encryption', 'penetration testing']
}

# Load your data
df = pd.read_csv('cleaned_job_data.csv')

# Function to categorize job titles 


def categorize_job_title(row):
    if row['Job Title'].lower() != 'software developer':
        return row['Job Title']

    skills = row['Key Skills'].lower()

    # Check each category
    for category, keywords in skill_categories.items():
        for keyword in keywords:
            if keyword in skills:
                return category

    # Default if no match found
    return 'Software Developer'


# Apply the categorization
df['Job Title'] = df.apply(categorize_job_title, axis=1)
df.to_csv('categorized_jobs.csv', index=False)
