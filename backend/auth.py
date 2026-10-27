"""
NeuroVQA — Doctor Authentication Service
Dual-Mode: Cloud Supabase Auth + Local SQLite JWT Fallback.
"""

from datetime import datetime, timedelta
import os
from typing import Optional, Union, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt

from .database import get_db, User, AuditLog
from .supabase_client import (
    supabase_signup,
    supabase_login,
    supabase_get_user,
    is_supabase_enabled,
    supabase_insert_audit_log,
)

SECRET_KEY = os.getenv("NEUROVQA_SECRET_KEY", "neurovqa-offline-clinical-secret-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class UserSignup(BaseModel):
    full_name: str
    email: str
    password: str
    hospital: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class UserOut(BaseModel):
    id: Union[int, str]
    full_name: str
    email: str
    specialty: Optional[str] = "Radiology / Neurology"

    class Config:
        from_attributes = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1. Try Supabase user validation
    if is_supabase_enabled():
        supa_user = await supabase_get_user(token)
        if supa_user:
            return supa_user

    # 2. Fallback to local JWT decode
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        user = db.query(User).filter(User.id == int(user_id)).first()
        if user:
            return {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "specialty": "Radiology / Neurology",
            }
    except Exception:
        pass

    raise credentials_exception


@router.post("/signup", response_model=Token)
async def signup(user_in: UserSignup, db: Session = Depends(get_db)):
    # 1. If Supabase is enabled, attempt Supabase Auth signup
    if is_supabase_enabled():
        supa_res = await supabase_signup(
            email=user_in.email,
            password=user_in.password,
            full_name=user_in.full_name,
            specialty=user_in.hospital or "Radiology / Neurology",
        )
        if supa_res and "access_token" in supa_res:
            await supabase_insert_audit_log({
                "action": "DOCTOR_SIGNUP",
                "metadata": {"email": user_in.email, "provider": "supabase"},
            }, access_token=supa_res["access_token"])
            return {"access_token": supa_res["access_token"], "token_type": "bearer"}

        # If signup succeeded but email confirmation is on or session returned separately, log in
        if supa_res:
            login_res = await supabase_login(user_in.email, user_in.password)
            if login_res and "access_token" in login_res:
                return {"access_token": login_res["access_token"], "token_type": "bearer"}

    # 2. Local SQLite Fallback
    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A doctor account with this email address already exists.",
        )

    hashed_pw = get_password_hash(user_in.password)
    user = User(
        full_name=user_in.full_name,
        email=user_in.email,
        hashed_password=hashed_pw,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Log audit event
    log = AuditLog(user_id=user.id, action="DOCTOR_SIGNUP")
    db.add(log)
    db.commit()

    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=Token)
async def login(creds: UserLogin, db: Session = Depends(get_db)):
    # 1. If Supabase is enabled, attempt Supabase Auth login
    if is_supabase_enabled():
        supa_res = await supabase_login(creds.email, creds.password)
        if supa_res and "access_token" in supa_res:
            await supabase_insert_audit_log({
                "action": "DOCTOR_LOGIN",
                "metadata": {"email": creds.email, "provider": "supabase"},
            }, access_token=supa_res["access_token"])
            return {"access_token": supa_res["access_token"], "token_type": "bearer"}

    # 2. Local SQLite Fallback
    user = db.query(User).filter(User.email == creds.email).first()
    if not user or not verify_password(creds.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Log audit event
    log = AuditLog(user_id=user.id, action="DOCTOR_LOGIN")
    db.add(log)
    db.commit()

    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
async def read_current_user(current_user: Dict[str, Any] = Depends(get_current_user)):
    return current_user
