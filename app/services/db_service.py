from sqlalchemy.orm import Session
from app.models.resume_models import ResumeData

def save_resume_data(db: Session, data: dict):
    resume = ResumeData(**data)
    db.add(resume)
    db.commit()
    db.refresh(resume)
    return resume
