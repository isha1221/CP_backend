from fastapi import APIRouter, Path, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.parser_service import extract_resume_data
from app.services.db_service import save_resume_data
from app.db import get_db
from app.models.resume_models import ResumeCreate, ResumeData, ResumeResponse, ResumeUpdate
from app.services.ml_service import predict_top_careers
from app.routers.auth_routes import get_current_user
from app.models.auth_models import User, PlanType
import os
import uuid

router = APIRouter()

# Ensure the resumes directory exists
RESUME_DIR = "./resumes"
os.makedirs(RESUME_DIR, exist_ok=True)


def check_resume_limit(user: User):
    """
    Check if the user has exceeded their resume limit based on their plan type
    """
    if user.plan_type == PlanType.RP_BASIC and user.resume_count >= 5:
        raise HTTPException(
            status_code=403,
            detail="Basic plan users can only upload 5 resumes. Please upgrade your plan to upload more."
        )
    elif user.plan_type == PlanType.RP_MOD and user.resume_count >= 15:
        raise HTTPException(
            status_code=403,
            detail="Moderate plan users can only upload 15 resumes. Please upgrade to Pro plan for unlimited uploads."
        )
    # RP_PRO users have unlimited resume uploads
    return True


@router.post("/upload", response_model=ResumeResponse)
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    print("✅ File received")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported.")

    check_resume_limit(current_user)

    contents = await file.read()
    print("✅ File read complete")

    file_extension = os.path.splitext(file.filename)[1]
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(RESUME_DIR, file_name)
    try:
        with open(file_path, "wb") as f:
            f.write(contents)
        print("✅ File saved:", file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        extracted_data = extract_resume_data(contents)
        print("✅ Parsing complete:", extracted_data)

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail=f"Failed to parse resume: {str(e)}")

    try:
        saved = save_resume_data(db, extracted_data, user_id=current_user.id)
        # Increment the resume count
        current_user.resume_count += 1
        db.commit()
        print("✅ Saved to DB")
        return saved
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save resume to database: {str(e)}")


@router.get("/{resume_id}", response_model=ResumeResponse)
def get_resume_by_id(
    resume_id: int = Path(..., description="The ID of the resume to retrieve"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    resume = db.query(ResumeData).filter(ResumeData.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    if resume.user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this resume")

    resume.skills = resume.skills.split(", ") if resume.skills else []
    return resume


@router.put("/{resume_id}", response_model=ResumeResponse)
def update_resume(
    resume_id: int,
    data: ResumeUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    resume = db.query(ResumeData).filter(ResumeData.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    if resume.user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to update this resume")

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


@router.post("/manual-update", response_model=ResumeResponse)
def manual_update(
    data: ResumeCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check resume limit
    check_resume_limit(current_user)

    resume_data = data.dict(exclude_unset=True)
    # Use a placeholder file_path (since no PDF is uploaded)
    file_path = "manual_entry.pdf"
    try:
        saved = save_resume_data(
            db, resume_data, user_id=current_user.id, file_path=file_path)
        # Increment resume_count after successful save
        current_user.resume_count += 1
        db.commit()
        return saved
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to save resume: {str(e)}")


@router.post("/predict-career/{resume_id}")
def predict_career(
    resume_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    resume = db.query(ResumeData).filter(ResumeData.id == resume_id).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    if resume.user_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this resume")

    skills = resume.skills.split(", ") if resume.skills else []
    result = predict_top_careers(skills)
    return {
        "top_predictions": result["top_predictions"],
        "probabilities": result["probabilities"]
    }
