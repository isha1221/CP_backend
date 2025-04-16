from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.db import Base, engine
# from app.models.resume_models import ResumeData
from app.routers import resume_routes
# auth_routes, resume_routes, prediction_routes, user_routes
from fastapi.responses import JSONResponse
from typing import Optional
from app.db import get_db
from sqlalchemy.orm import Session
from app.services.db_service import save_resume_data
from app.models.resume_models import ResumeResponse

app = FastAPI()

# Allow frontend communication (e.g. React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"], #allow all existing HTTP methods
    allow_headers=["*"], #Allow all custom headers to be sent in the request. 
)


app.include_router(resume_routes.router, prefix="/resume")


# --- Health Check ---
@app.get("/")
def root():
    return {"message": "Resume Parser & Career Recommender API is running."}


# --- Generate PDF ---
@app.get("/generate-report/{user_id}")
def generate_report(user_id: int):
    # Generate PDF using WeasyPrint
    return JSONResponse({"pdf_url": f"/reports/report_{user_id}.pdf"})

Base.metadata.create_all(bind=engine)




