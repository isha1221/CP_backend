# from fastapi import APIRouter, Path, UploadFile, File, HTTPException, Depends
# from sqlalchemy.orm import Session
# from app.services.parser_service import extract_resume_data
# from app.services.db_service import save_resume_data
# from app.db import get_db
# from app.models.resume_models import ResumeCreate, ResumeData, ResumeResponse, ResumeUpdate
# from app.services.ml_service import predict_top_careers
# from app.routers.auth_routes import get_current_user
# from app.models.auth_models import User, PlanType
# import os
# import uuid
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query
from sqlalchemy.orm import Session
from typing import List
import os
import uuid
from datetime import datetime
from app.services.ml_service import predict_top_careers
from app.models.auth_models import User, PlanType
from app.models.resume_models import ResumeData, ResumeResponse, ResumeListItem,ResumeUpdate,ResumeCreate
from app.db import get_db
from app.routers.auth_routes import get_current_user
from app.services.parser_service import extract_resume_data
from app.services.db_service import save_resume_data

router = APIRouter(tags=["Resumes"])

# Ensure the resumes directory exists
RESUME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
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
# Add this to your resume_routes.py file

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query
from fastapi.responses import FileResponse
from app.services.file_service import get_file_path

# ... existing imports and code ...

@router.get("/{resume_id}/download")
async def download_resume(
    resume_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Download a resume PDF file
    """
    resume = db.query(ResumeData).filter(
        ResumeData.id == resume_id,
        ResumeData.user_id == current_user.id
    ).first()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    file_path = get_file_path(resume.stored_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Resume file not found")
    
    # Return the file as a download with its original filename
    return FileResponse(
        path=file_path, 
        filename=resume.original_filename,
        media_type="application/pdf"
    )

@router.post("/upload", response_model=ResumeResponse)
async def upload_resume(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    print("✅ File received")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    check_resume_limit(current_user)

    contents = await file.read()
    print("✅ File read complete")

    file_extension = os.path.splitext(file.filename)[1]
    stored_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(RESUME_DIR, stored_filename)
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
        extracted_data["original_filename"] = file.filename
        extracted_data["stored_filename"] = stored_filename
        extracted_data["upload_date"] = datetime.utcnow()

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

@router.get("/", response_model=List[ResumeListItem])
async def get_resumes(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get a list of all resumes uploaded by the current user
    """
    resumes = db.query(ResumeData).filter(
        ResumeData.user_id == current_user.id
    ).order_by(
        ResumeData.upload_date.desc()
    ).offset(skip).limit(limit).all()
    
    return resumes
# @router.get("/{resume_id}", response_model=ResumeResponse)
# def get_resume_by_id(
#     resume_id: int = Path(..., description="The ID of the resume to retrieve"),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     resume = db.query(ResumeData).filter(ResumeData.id == resume_id).first()
#     if not resume:
#         raise HTTPException(status_code=404, detail="Resume not found")

#     if resume.user_id != current_user.id:
#         raise HTTPException(
#             status_code=403, detail="Not authorized to access this resume")

#     resume.skills = resume.skills.split(", ") if resume.skills else []
#     return resume
@router.get("/{resume_id}", response_model=ResumeResponse)
async def get_resume(
    resume_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific resume
    """
    resume = db.query(ResumeData).filter(
        ResumeData.id == resume_id,
        ResumeData.user_id == current_user.id
    ).first()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Convert skills string back to list
    if resume.skills:
        resume.skills = resume.skills.split(", ")
    else:
        resume.skills = []
        
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
    
@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a resume from the database and remove the file
    """
    resume = db.query(ResumeData).filter(
        ResumeData.id == resume_id,
        ResumeData.user_id == current_user.id
    ).first()
    
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    
    # Delete the file
    file_path = os.path.join(RESUME_DIR, resume.stored_filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Delete from database and update user's resume count
    db.delete(resume)
    # current_user.resume_count -= 1
    db.commit()
    
    return {"message": "Resume deleted successfully"}


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
