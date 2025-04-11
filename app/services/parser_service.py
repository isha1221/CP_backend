import fitz  # PyMuPDF
import re
from typing import Dict


def extract_resume_data(file_bytes: bytes) -> Dict:
    # Load PDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    print("Extracted text:", text)
    # Basic regex-based parsing (can be extended)
    # name_match = re.search(r"(Name[:\s]+)([A-Za-z\s]+)", text, re.IGNORECASE)
    name_match = re.search(r"(?:Name|Full Name|Candidate Name)[:\s]+([A-Za-z\s]+)", text, re.IGNORECASE)
    if not name_match:
        # Fallback: Capture only the first two capitalized words at the start of the document
        name_match = re.search(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s|\n)", text, re.MULTILINE)
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    phone_match = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
    # skills_match = re.findall(r"\b(Python|Java|React|Node|SQL|ML|AI|Django|FastAPI)\b", text, re.IGNORECASE)
    
    skills_text = ""
    skills_section_match = re.search(r"(?:SKILLS|TECHNICAL SKILLS|COMPETENCIES)[:\s]*(.*?)(?=\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
    if skills_section_match:
        skills_text = skills_section_match.group(1).strip()

    # Match skills (capitalized words or compound terms) within the skills section
    skills_match = re.findall(r"\b([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)?(?:[A-Z][a-zA-Z]*)*)\b", skills_text, re.IGNORECASE)

    # Heuristic filtering to include only likely skills, without a fixed skill_patterns set
    def is_skill(term):
        term_lower = term.lower()
        # Exclude terms with digits (e.g., CGPA, 2023)
        if any(c.isdigit() for c in term):
            return False
        # Exclude common degree abbreviations
        degree_patterns = {'b.tech', 'm.tech', 'bsc', 'msc', 'phd'}
        if term_lower in degree_patterns:
            return False
        # Exclude short terms unlikely to be skills
        if len(term) <= 2:
            return False
        # Exclude months
        months = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}
        if term_lower in months:
            return False
        # Exclude common non-skill words
        non_skills = {
            'achievements', 'education', 'journal', 'published', 'analysis', 'research', 'usage',
            'using', 'international', 'water', 'sensor', 'data', 'paper', 'flow', 'science', 'structures','instrumentation'
        }
        if term_lower in non_skills:
            return False
        # Heuristic: Include terms with dots (e.g., React.js) or multiple capitals (e.g., GoLang) as likely skills
        if '.' in term_lower or (len(re.findall(r'[A-Z]', term)) > 1 and len(term) > 3):
            return True
        # Fallback: Include terms longer than 3 characters that start with a capital and aren't in non_skills
        return len(term) > 3 and term[0].isupper()

    skills = list(set([s.capitalize() for s in skills_match if is_skill(s)]))

    # Extract name
    name = name_match.group(1).strip() if name_match and name_match.group(1) else None
        
    name = name_match.group(1).strip() if name_match and name_match.group(1) else None
    return {
        "name": name,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "skills": skills
        # "skills": ", ".join(set(skills_match)) if skills_match else None
    }
