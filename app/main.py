from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.db import Base, engine
from app.routers import resume_routes
from fastapi.responses import JSONResponse
from app.routers import auth_routes


origins = ["http://localhost:3000"]


app = FastAPI()

# Allow frontend communication (e.g. React)
app.add_middleware(
    CORSMiddleware,
    # Set your frontend domain in production
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # allow all existing HTTP methods
    allow_headers=["*"],  # Allow all custom headers to be sent in the request.
)


app.include_router(resume_routes.router, prefix="/resume")
app.include_router(auth_routes.router, prefix="/auth", tags=["auth"])


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
