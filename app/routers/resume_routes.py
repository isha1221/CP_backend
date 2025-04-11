from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.parser_service import extract_resume_data
from app.models.resume_models import ResumeCreate


router = APIRouter()

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    contents = await file.read()
    extracted_data = extract_resume_data(contents)

    return {
        "filename": file.filename,
        "parsed_data": extracted_data
    }


@router.post("/manual-update")
def manual_update(data: ResumeCreate):
    # Cleaned and validated data is available in `data`
    updated_resume = {
        "name": data.name,
        "email": data.email,
        "phone": data.phone,
        "skills": data.skills or [],
        "experience":data.experience,
    }

    # You can store this in DB or pass to ML model
    return {
        "message": "Resume data updated successfully.",
        "updated_data": updated_resume
    }