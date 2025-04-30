from sqlalchemy.orm import Session
from app.models.resume_models import ResumeData

# def save_resume_data(db: Session, data: dict):
#     if isinstance(data.get("skills"), list):
#         data["skills"] = ", ".join(data["skills"]) 
#     resume = ResumeData(**data)
#     db.add(resume)
#     db.commit()
#     db.refresh(resume)
    
#     # important
#     resume.skills = resume.skills.split(", ") if resume.skills else []
#     return resume

def save_resume_data(db: Session, data: dict, user_id: int):
    if isinstance(data.get("skills"), list):
        data["skills"] = ", ".join(data["skills"])
    
    resume = ResumeData(**data, user_id=user_id)  # ✅ Add user_id
    db.add(resume)
    db.commit()
    db.refresh(resume)
    
    resume.skills = resume.skills.split(", ") if resume.skills else []
    return resume
