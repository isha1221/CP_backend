import fitz  # PyMuPDF
import re
from typing import Dict


def extract_resume_data(file_bytes: bytes) -> Dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    # Basic extractions
    name_match = re.search(
        r"(?:Name|Full Name|Candidate Name)[:\s]+([A-Za-z\s]+)", text, re.IGNORECASE)
    if not name_match:
        name_match = re.search(
            r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s|\n)", text, re.MULTILINE)

    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    phone_match = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)

    # Extract experience section
    experience_text = ""
    experience_pattern = re.compile(r"(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE)[:\s]*(?:\n|\r\n)(.*?)(?=(?:EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS)[:\s]*(?:\n|\r\n)|\Z)",
                                    re.IGNORECASE | re.DOTALL)
    experience_match = experience_pattern.search(text)
    if experience_match:
        experience_text = experience_match.group(1).strip()

    # Extract skills - Improved approach for categorized skills
    skills = []
    # First try to find a skills section
    section_headers = ["EDUCATION", "PROJECTS", "CERTIFICATIONS",
                       "CERTIFICATES", "EXPERIENCE", "REFERENCES"]
    section_pattern = "|".join(section_headers)

    skills_pattern = re.compile(
        r"(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES)[:\s]*(?:\n|\r\n)(.*?)(?=(?:" +
        section_pattern + ")[:\s]*(?:\n|\r\n)|\Z)",
        re.IGNORECASE | re.DOTALL
    )

    skills_match = skills_pattern.search(text)
    if skills_match:
        skills_text = skills_match.group(1).strip()

        # Handle categorized skills format (● Category: Skill1, Skill2)
        categorized_skills = re.findall(
            r'●[^:]+:(.*?)(?=●|\Z)', skills_text, re.DOTALL)
        if categorized_skills:
            for category in categorized_skills:
                # Extract skills from each category
                category_skills = re.findall(r'([^,●\n]+)', category)
                for skill in category_skills:
                    skill = skill.strip()
                    if skill and not any(s.lower() == skill.lower() for s in skills):
                        skills.append(skill)
        else:
            # Fallback to previous parsing methods if not in categorized format
            if '|' in skills_text:
                # Skills separated by pipe character
                raw_skills = [s.strip()
                              for s in re.split(r'\s*\|\s*', skills_text)]
            elif '•' in skills_text or '●' in skills_text:
                # Skills in bullet points (handle both bullet types)
                bullet_pattern = r'\s*[•●]\s*'
                raw_skills = [s.strip()
                              for s in re.split(bullet_pattern, skills_text)]
            elif ',' in skills_text:
                # Comma separated skills
                raw_skills = [s.strip()
                              for s in re.split(r'\s*,\s*', skills_text)]
            else:
                # Skills on separate lines
                raw_skills = [line.strip()
                              for line in skills_text.splitlines()]

            # Filter out empty items and common non-skills
            blacklist = {
                'innovation in user', 'development', 'usage', 'projects', 'experience',
                'communication', 'design', 'international', 'published', 'research', 'paper'
            }

            for skill in raw_skills:
                skill = skill.strip()
                if (skill and
                    not any(c.isdigit() for c in skill) and
                    not any(skill.lower().startswith(b) for b in blacklist) and
                        not any(s.lower() == skill.lower() for s in skills)):
                    skills.append(skill)

    name = name_match.group(1).strip(
    ) if name_match and name_match.group(1) else None

    return {
        "name": name,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "skills": skills,
        "experience": experience_text
    }
