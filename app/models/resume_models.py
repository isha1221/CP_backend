# from pydantic import BaseModel, EmailStr
# from typing import List, Optional

# class ResumeUpdate(BaseModel):
#     name: Optional[str]
#     email: Optional[EmailStr]
#     phone: Optional[str]
#     skills: Optional[List[str]]


# app/models/resume_model.py
from sqlalchemy import Column, Integer, String, Text
from app.db import Base
from typing import List, Optional
from pydantic import BaseModel

class ResumeData(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    skills = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)

class ResumeBase(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    skills: Optional[List[str]]
    experience: Optional[str]

class ResumeCreate(ResumeBase):
    pass


class ResumeUpdate(ResumeBase):
    pass

class ResumeResponse(ResumeBase):
    id: int

    class Config:
        orm_mode = True  
        
        