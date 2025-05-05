# app/models/resume_model.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from app.db import Base
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class ResumeData(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    skills = Column(Text, nullable=True)
    experience = Column(Text, nullable=True)
    
    # New fields for file information
    original_filename = Column(String, nullable=True)
    stored_filename = Column(String, nullable=False)  # UUID-based filename
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

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
    original_filename: Optional[str]
    upload_date: Optional[datetime]

    class Config:
        orm_mode = True

class ResumeListItem(BaseModel):
    id: int
    name: Optional[str]
    original_filename: str
    upload_date: datetime
    
    class Config:
        orm_mode = True