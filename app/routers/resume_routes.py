
from fastapi import APIRouter, Path, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.parser_service import extract_resume_data
from app.services.db_service import save_resume_data
from app.db import get_db
from app.models.resume_models import ResumeCreate, ResumeData, ResumeResponse, ResumeUpdate
from app.services.ml_service import predict_top_careers
from app.routers.auth_routes import get_current_user
from app.models.auth_models import User 

router = APIRouter()

@router.post("/upload", response_model=ResumeResponse)
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    print("✅ File received")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    contents = await file.read()
    print("✅ File read complete")
    extracted_data = extract_resume_data(contents)
    print("✅ Parsing complete:", extracted_data)

    # Save parsed resume data to DB with user_id from JWT token
    saved = save_resume_data(db, extracted_data, user_id=current_user.id)
    print("✅ Saved to DB")

    return saved

@router.get("/{resume_id}", response_model=ResumeResponse)
def get_resume_by_id(
    resume_id: int = Path(..., description="The ID of the resume to retrieve"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    resume = db.query(ResumeData).filter(ResumeData.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Check if the resume belongs to the authenticated user
    if resume.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resume")

    # Convert comma-separated string back to list for skills
    resume.skills = resume.skills.split(", ") if resume.skills else []
    
    return resume

@router.put("/{resume_id}", response_model=ResumeResponse)
def update_resume(
    resume_id: int,
    data: ResumeUpdate,
    db: Session = Depends(get_db)
):
    resume = db.query(ResumeData).filter(ResumeData.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    if data.name is not None:
        resume.name = data.name
    if data.email is not None:
        resume.email = data.email
    if data.phone is not None:
        resume.phone = data.phone
    if data.experience is not None:
        resume.experience = data.experience
    if data.skills is not None:
        resume.skills = ", ".join(data.skills)

    db.commit()
    db.refresh(resume)
    resume.skills = resume.skills.split(", ") if resume.skills else []

    return resume

@router.post("/manual-update")
def manual_update(data: ResumeCreate):
    # Cleaned and validated data is available in `data`
    updated_resume = {
        "name": data.name,
        "email": data.email,
        "phone": data.phone,
        "skills": data.skills or [],
        "experience": data.experience,
    }

    # You can store this in DB or pass to ML model
    return {
        "message": "Resume data updated successfully.",
        "updated_data": updated_resume
    }
    

@router.post("/predict-career/{resume_id}")
def predict_career(resume_id: int, db: Session = Depends(get_db)):
    resume = db.query(ResumeData).filter(ResumeData.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    skills = resume.skills.split(", ") if resume.skills else []

    result = predict_top_careers(skills)

    return {
        "top_predictions": result["top_predictions"],
        "probabilities": result["probabilities"]
    }
