# import fitz  # PyMuPDF
# import re
# from typing import Dict


# def extract_resume_data(file_bytes: bytes) -> Dict:
#     # Load PDF
#     doc = fitz.open(stream=file_bytes, filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()

#     print("Extracted text:", text)
#     # Basic regex-based parsing (can be extended)
#     # name_match = re.search(r"(Name[:\s]+)([A-Za-z\s]+)", text, re.IGNORECASE)
#     name_match = re.search(r"(?:Name|Full Name|Candidate Name)[:\s]+([A-Za-z\s]+)", text, re.IGNORECASE)
#     if not name_match:
#         # Fallback: Capture only the first two capitalized words at the start of the document
#         name_match = re.search(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s|\n)", text, re.MULTILINE)
#     email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
#     phone_match = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
#     # skills_match = re.findall(r"\b(Python|Java|React|Node|SQL|ML|AI|Django|FastAPI)\b", text, re.IGNORECASE)
    
#     skills_text = ""
#     skills_section_match = re.search(r"(?:SKILLS|TECHNICAL SKILLS|COMPETENCIES)[:\s]*(.*?)(?=\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
#     if skills_section_match:
#         skills_text = skills_section_match.group(1).strip()

#     # Match skills (capitalized words or compound terms) within the skills section
#     skills_match = re.findall(r"\b([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)?(?:[A-Z][a-zA-Z]*)*)\b", skills_text, re.IGNORECASE)

#     # Heuristic filtering to include only likely skills, without a fixed skill_patterns set
#     def is_skill(term):
#         term_lower = term.lower()
#         # Exclude terms with digits (e.g., CGPA, 2023)
#         if any(c.isdigit() for c in term):
#             return False
#         # Exclude common degree abbreviations
#         degree_patterns = {'b.tech', 'm.tech', 'bsc', 'msc', 'phd'}
#         if term_lower in degree_patterns:
#             return False
#         # Exclude short terms unlikely to be skills
#         if len(term) <= 2:
#             return False
#         # Exclude months
#         months = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}
#         if term_lower in months:
#             return False
#         # Exclude common non-skill words
#         non_skills = {
#             'achievements', 'education', 'journal', 'published', 'analysis', 'research', 'usage',
#             'using', 'international', 'water', 'sensor', 'data', 'paper', 'flow', 'science', 'structures','instrumentation'
#         }
#         if term_lower in non_skills:
#             return False
#         # Heuristic: Include terms with dots (e.g., React.js) or multiple capitals (e.g., GoLang) as likely skills
#         if '.' in term_lower or (len(re.findall(r'[A-Z]', term)) > 1 and len(term) > 3):
#             return True
#         # Fallback: Include terms longer than 3 characters that start with a capital and aren't in non_skills
#         return len(term) > 3 and term[0].isupper()

#     skills = list(set([s.capitalize() for s in skills_match if is_skill(s)]))

#     # Extract name
#     name = name_match.group(1).strip() if name_match and name_match.group(1) else None
        
#     name = name_match.group(1).strip() if name_match and name_match.group(1) else None
#     return {
#         "name": name,
#         "email": email_match.group(0) if email_match else None,
#         "phone": phone_match.group(0) if phone_match else None,
#         "skills": skills
#         # "skills": ", ".join(set(skills_match)) if skills_match else None
#     }


# # import fitz  # PyMuPDF
# # import re
# # from typing import Dict

# # def extract_resume_data(file_bytes: bytes) -> Dict:
# #     doc = fitz.open(stream=file_bytes, filetype="pdf")
# #     text = ""
# #     for page in doc:
# #         text += page.get_text()
# #     # print("Extracted text:", text)  # Uncomment for debugging if needed

# #     # Extract name, email, phone
# #     name_match = re.search(r"(?:Name|Full Name|Candidate Name)[:\s]+([A-Za-z\s]+)", text, re.IGNORECASE)
# #     if not name_match:
# #         name_match = re.search(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s|\n)", text, re.MULTILINE)
# #     email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
# #     phone_match = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)

# #     # === Extract SKILLS: Capture lines after 'SKILLS' until next section ===
# #     skills = []
# #     # skills_block_match = re.search(r"(?:SKILLS|Technical\s+Skills|Core\s+Competencies)[\s:\-]*(?:[\n\r]+)((?:[^\n]*[\n\r]?)+?)(?=(?:CERTIFICATES|PROJECTS|EDUCATION|PROFESSIONAL EXPERIENCE)[\s:\-]*[\n\r]|$)", text, re.IGNORECASE | re.MULTILINE)
# #     # skills_block_match = re.search(r"(?:SKILLS|Technical\s+Skills|Core\s+Competencies)[\s:\-]*(?:[\n\r]+)((?:[^\n]*[\n\r]?)+?)(?=(?:CERTIFICATIONS|PROJECTS|EDUCATION|PROFESSIONAL EXPERIENCE)[\s:\-]*[\n\r])", text, re.IGNORECASE | re.MULTILINE)
# #     skills_block_match = re.search(r"(?:SKILLS|Technical\s+Skills|Core\s+Competencies)[\s:\-]*(?:[\n\r]+)((?:(?!CERTIFICATES|CERTIFICATIONS:)[^\n]*[\n\r]?)+)(?=(?:CERTIFICATIONS|PROJECTS|EDUCATION|PROFESSIONAL EXPERIENCE)[\s:\-]*[\n\r])", text, re.IGNORECASE | re.MULTILINE)
# #     if skills_block_match:
# #         skills_block = skills_block_match.group(1).strip()
# #         # print("Skills block:", skills_block)  # Uncomment for debugging
# #         # Split into lines and filter non-empty skills
# #         if '|' in skills_block:
# #             # Split inline skills by | and clean each segment
# #             skill_lines = [s.strip() for s in re.split(r'\s*\|\s*', skills_block) if s.strip()]
# #         else:    
# #             skill_lines = [line.strip() for line in skills_block.splitlines() if line.strip()]
        
# #         def clean_skill(s):
# #             s = s.strip()
# #             # if not s or any(c.isdigit() for c in s):
# #             #     return None
# #             if not s or s.startswith("CERTIFICATIONS:") or any(c.isdigit() for c in s):
# #                 return None
# #             blacklist = {
# #                 'innovation in user', 'development', 'usage', 'projects', 'experience',
# #                 'communication', 'design', 'international', 'published', 'research', 'paper'
# #             }
# #             if any(s.lower().startswith(b) for b in blacklist):
# #                 return None
# #             return s

# #         skills = list(filter(None, [clean_skill(s) for s in skill_lines]))

# #     name = name_match.group(1).strip() if name_match and name_match.group(1) else None

# #     return {
# #         "name": name,
# #         "email": email_match.group(0) if email_match else None,
# #         "phone": phone_match.group(0) if phone_match else None,
# #         "skills": skills
# #     }
import fitz  # PyMuPDF
import re
from typing import Dict

def extract_resume_data(file_bytes: bytes) -> Dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Basic extractions
    name_match = re.search(r"(?:Name|Full Name|Candidate Name)[:\s]+([A-Za-z\s]+)", text, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s|\n)", text, re.MULTILINE)
    
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
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
    section_headers = ["EDUCATION", "PROJECTS", "CERTIFICATIONS", "CERTIFICATES", "EXPERIENCE", "REFERENCES"]
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
        categorized_skills = re.findall(r'●[^:]+:(.*?)(?=●|\Z)', skills_text, re.DOTALL)
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
                raw_skills = [s.strip() for s in re.split(r'\s*\|\s*', skills_text)]
            elif '•' in skills_text or '●' in skills_text:
                # Skills in bullet points (handle both bullet types)
                bullet_pattern = r'\s*[•●]\s*'
                raw_skills = [s.strip() for s in re.split(bullet_pattern, skills_text)]
            elif ',' in skills_text:
                # Comma separated skills
                raw_skills = [s.strip() for s in re.split(r'\s*,\s*', skills_text)]
            else:
                # Skills on separate lines
                raw_skills = [line.strip() for line in skills_text.splitlines()]
            
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
    
    name = name_match.group(1).strip() if name_match and name_match.group(1) else None
    
    return {
        "name": name,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "skills": skills,
        "experience": experience_text
    }

# import fitz  # PyMuPDF
# import re
# from typing import Dict

# def extract_resume_data(file_bytes: bytes) -> Dict:
#     doc = fitz.open(stream=file_bytes, filetype="pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
    
#     # Basic extractions
#     name_match = re.search(r"(?:Name|Full Name|Candidate Name)[:\s]+([A-Za-z\s]+)", text, re.IGNORECASE)
#     if not name_match:
#         name_match = re.search(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?=\s|\n)", text, re.MULTILINE)
    
#     email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
#     phone_match = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
    
#     # Extract experience section
#     experience_text = ""
#     experience_pattern = re.compile(r"(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE)[:\s]*(?:\n|\r\n)(.*?)(?=(?:EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS)[:\s]*(?:\n|\r\n)|\Z)", 
#                                   re.IGNORECASE | re.DOTALL)
#     experience_match = experience_pattern.search(text)
#     if experience_match:
#         experience_text = experience_match.group(1).strip()
    
#     # Extract skills - Simplified approach
#     skills = []
#     # First try to find a skills section
#     section_headers = ["EDUCATION", "PROJECTS", "CERTIFICATIONS", "CERTIFICATES", "EXPERIENCE", "REFERENCES"]
#     section_pattern = "|".join(section_headers)
    
#     skills_pattern = re.compile(
#         r"(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES)[:\s]*(?:\n|\r\n)(.*?)(?=(?:" + 
#         section_pattern + ")[:\s]*(?:\n|\r\n)|\Z)", 
#         re.IGNORECASE | re.DOTALL
#     )
    
#     skills_match = skills_pattern.search(text)
#     if skills_match:
#         skills_text = skills_match.group(1).strip()
        
#         # Parse skills from the section
#         if '|' in skills_text:
#             # Skills separated by pipe character
#             raw_skills = [s.strip() for s in re.split(r'\s*\|\s*', skills_text)]
#         elif '•' in skills_text:
#             # Skills in bullet points
#             raw_skills = [s.strip() for s in re.split(r'\s*•\s*', skills_text)]
#         elif ',' in skills_text:
#             # Comma separated skills
#             raw_skills = [s.strip() for s in re.split(r'\s*,\s*', skills_text)]
#         else:
#             # Skills on separate lines
#             raw_skills = [line.strip() for line in skills_text.splitlines()]
        
#         # Filter out empty items and common non-skills
#         blacklist = {
#             'innovation in user', 'development', 'usage', 'projects', 'experience',
#             'communication', 'design', 'international', 'published', 'research', 'paper'
#         }
        
#         for skill in raw_skills:
#             skill = skill.strip()
#             if (skill and 
#                 not any(c.isdigit() for c in skill) and 
#                 not any(skill.lower().startswith(b) for b in blacklist)):
#                 skills.append(skill)
    
#     name = name_match.group(1).strip() if name_match and name_match.group(1) else None
    
#     return {
#         "name": name,
#         "email": email_match.group(0) if email_match else None,
#         "phone": phone_match.group(0) if phone_match else None,
#         "skills": skills,
#         "experience": experience_text
#     }

