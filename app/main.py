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
from app.schemas import ResumeResponse

app = FastAPI()

# Allow frontend communication (e.g. React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"], #allow all existing HTTP methods
    allow_headers=["*"], #Allow all custom headers to be sent in the request. 
)

#Add routes
# app.include_router(auth_routes.router,prefix="/auth")
# app.include_router(user_routes.router, prefix="/user")
app.include_router(resume_routes.router, prefix="/resume")
# app.include_router(prediction_routes.router, prefix="/predict")

# --- Health Check ---
@app.get("/")
def root():
    return {"message": "Resume Parser & Career Recommender API is running."}


# --- Upload Resume ---
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    # You will use PyMuPDF + regex here
    return {"filename": file.filename, "status": "Resume uploaded & parsed (dummy response)"}


# --- Manual Data Input ---
@app.post("/manual-entry")
def manual_entry(
    name: str = Form(...),
    email: str = Form(...),
    skills: str = Form(...),
    experience: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):  
    data = {
        "name": name,
        "email": email,
        "skills": skills,
        "experience": experience
    }
    saved = save_resume_data(db, data)
    # Save data to DB
    return {"status": "Manual data saved",
            "id": saved.id,
            "name": name}


# --- Get Career Recommendation ---
@app.post("/predict-career")
def predict_career(user_id: int):
    # Load resume features from DB by user_id
    # Run ML model prediction
    return {"recommended_career": "Data Scientist"}


# --- Generate PDF ---
@app.get("/generate-report/{user_id}")
def generate_report(user_id: int):
    # Generate PDF using WeasyPrint
    return JSONResponse({"pdf_url": f"/reports/report_{user_id}.pdf"})

Base.metadata.create_all(bind=engine)





# from typing import Union

# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()


# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: Union[bool, None] = None


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}
# from typing import Union
# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()

# class Info(BaseModel):
#     name:str
#     price:float

# @app.get("/")
# def read_root():
#     return {"Hello, 1st api"}


# @app.get("/greeting/{name}/{num}")
# def read_name(name:str,num:int):
#     # return {"hello "+ name+"/n"+"my"+num+"api"}
#     return{f"hello {name} my {num} api"} 