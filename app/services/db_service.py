from sqlalchemy.orm import Session
from app.models.resume_models import ResumeData



def save_resume_data(db: Session, data: dict, user_id: int):
    if isinstance(data.get("skills"), list):
        data["skills"] = ", ".join(data["skills"])
    
    # Create a copy of the data dict to avoid unexpected key errors
    resume_data = {
        "user_id": user_id
    }
    
    # Only add keys that exist in ResumeData model
    valid_keys = ["name", "email", "phone", "skills", "experience"]
    for key in valid_keys:
        if key in data:
            resume_data[key] = data[key]
    
    resume = ResumeData(**resume_data)
    db.add(resume)
    db.flush()  # Flush to get the ID but don't commit yet
    
    # Convert skills back to list for the response
    response_dict = {
        "id": resume.id,
        "name": resume.name,
        "email": resume.email,
        "phone": resume.phone,
        "experience": resume.experience,
        "skills": resume.skills.split(", ") if resume.skills else []
    }
    
    return response_dict


