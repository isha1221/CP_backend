# from sqlalchemy.orm import Session
# from app.models.resume_models import ResumeData



# def save_resume_data(db: Session, data: dict, user_id: int):
#     if isinstance(data.get("skills"), list):
#         data["skills"] = ", ".join(data["skills"])
    
#     # Create a copy of the data dict to avoid unexpected key errors
#     resume_data = {
#         "user_id": user_id
#     }
    
#     # Only add keys that exist in ResumeData model
#     valid_keys = ["name", "email", "phone", "skills", "experience"]
#     for key in valid_keys:
#         if key in data:
#             resume_data[key] = data[key]
    
#     resume = ResumeData(**resume_data)
#     db.add(resume)
#     db.flush()  # Flush to get the ID but don't commit yet
    
#     # Convert skills back to list for the response
#     response_dict = {
#         "id": resume.id,
#         "name": resume.name,
#         "email": resume.email,
#         "phone": resume.phone,
#         "experience": resume.experience,
#         "skills": resume.skills.split(", ") if resume.skills else []
#     }
    
#     return response_dict


from sqlalchemy.orm import Session
from app.models.resume_models import ResumeData

def save_resume_data(db: Session, data: dict, user_id: int):
    """
    Save resume data to the database
    """
    if isinstance(data.get("skills"), list):
        data["skills"] = ", ".join(data["skills"])
    
    # Create a copy of the data dict to avoid unexpected key errors
    resume_data = {
        "user_id": user_id
    }
    
    # Only add keys that exist in ResumeData model
    valid_keys = ["name", "email", "phone", "skills", "experience", 
                  "original_filename", "stored_filename", "upload_date"]
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
        "skills": resume.skills.split(", ") if resume.skills else [],
        "original_filename": resume.original_filename,
        "upload_date": resume.upload_date
    }
    
    return response_dict

def get_resume_count_by_user(db: Session, user_id: int) -> int:
    """
    Get the number of resumes uploaded by a user
    """
    return db.query(ResumeData).filter(ResumeData.user_id == user_id).count()

def get_resume_by_id(db: Session, resume_id: int, user_id: int):
    """
    Get a resume by ID, ensuring it belongs to the specified user
    """
    resume = db.query(ResumeData).filter(
        ResumeData.id == resume_id,
        ResumeData.user_id == user_id
    ).first()
    
    if resume and resume.skills:
        resume.skills = resume.skills.split(", ")
    elif resume:
        resume.skills = []
        
    return resume