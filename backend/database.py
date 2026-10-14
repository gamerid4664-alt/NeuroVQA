"""
NeuroVQA — SQLite Database Models and Session Configuration
Implements the 6 core schemas defined in NeuroVQA-design.md
"""

from datetime import datetime
import json
import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, 'neurovqa.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(128), nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    diagnoses = relationship("Diagnosis", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")
    feedbacks = relationship("DoctorFeedback", back_populates="user")


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    mri_file_path = Column(String(512), nullable=False)
    question = Column(String(512), nullable=False)
    answer = Column(String(256), nullable=False)
    confidence = Column(Float, nullable=False)
    top_3_predictions_json = Column(Text, nullable=False, default="[]")
    heatmap_path = Column(String(512), nullable=True)
    report_path = Column(String(512), nullable=True)
    model_version = Column(String(64), default="NeuroVQA-v26-BiomedCLIP-LoRA")
    latency_seconds = Column(Float, default=1.75)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="diagnoses")
    feedbacks = relationship("DoctorFeedback", back_populates="diagnosis")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"), nullable=True)
    title = Column(String(256), default="Clinical Consultation")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(32), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class DoctorFeedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    feedback_type = Column(String(64), nullable=False)  # correct | partially_correct | incorrect | uncertain
    doctor_note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    diagnosis = relationship("Diagnosis", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    diagnosis_id = Column(Integer, nullable=True)
    action = Column(String(64), nullable=False)
    model_version = Column(String(64), default="NeuroVQA-v26-BiomedCLIP-LoRA")
    model_checkpoint = Column(String(128), default="neurovqa_weights.pt")
    inference_latency = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
