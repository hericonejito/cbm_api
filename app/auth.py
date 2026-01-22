"""
OAuth2 Authentication Module with JWT Tokens

This module provides OAuth2 password flow authentication with JWT tokens
for the CBM API. Users are stored in a local JSON file.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel


# Configuration
# In production, use environment variables for these secrets
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-in-production-use-openssl-rand-hex-32")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Path to users file
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme - tokens are sent via Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    permissions: List[str] = []


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    permissions: List[str] = []


class UserInDB(User):
    hashed_password: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    full_name: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


# User database functions
def load_users_db() -> dict:
    """Load users from JSON file."""
    if not os.path.exists(USERS_FILE):
        return {"users": {}}

    with open(USERS_FILE, 'r') as f:
        return json.load(f)


def save_users_db(db: dict) -> None:
    """Save users to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(db, f, indent=2)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database by username."""
    db = load_users_db()
    users = db.get("users", {})

    if username in users:
        user_data = users[username]
        return UserInDB(**user_data)
    return None


# Password functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    # Bcrypt has a 72-byte limit, truncate if necessary
    # Only truncate if password exceeds 72 bytes (very rare)
    if isinstance(plain_password, str):
        password_bytes = plain_password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to 72 bytes, then decode back to string
            plain_password = password_bytes[:72].decode('utf-8', errors='ignore')
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # Log the error for debugging (in production, use proper logging)
        print(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password."""
    # Bcrypt has a 72-byte limit, truncate if necessary
    # Only truncate if password exceeds 72 bytes (very rare)
    if isinstance(password, str):
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to 72 bytes, then decode back to string
            password = password_bytes[:72].decode('utf-8', errors='ignore')
    return pwd_context.hash(password)


# Authentication functions
def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user by username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Dependency functions for route protection
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Dependency to get the current authenticated user from JWT token.

    Usage:
        @app.get("/protected")
        async def protected_route(current_user: User = Depends(get_current_user)):
            return {"user": current_user.username}
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        permissions: List[str] = payload.get("permissions", [])
        token_data = TokenData(username=username, permissions=permissions)
    except JWTError:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        permissions=user.permissions
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to get current user and verify they are not disabled.

    Usage:
        @app.get("/protected")
        async def protected_route(current_user: User = Depends(get_current_active_user)):
            return {"user": current_user.username}
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    return current_user


def require_permissions(*required_permissions: str):
    """
    Dependency factory to require specific permissions.

    Usage:
        @app.post("/admin-only")
        async def admin_route(current_user: User = Depends(require_permissions("admin"))):
            return {"user": current_user.username}
    """
    async def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        for permission in required_permissions:
            if permission not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied. Required: {required_permissions}"
                )
        return current_user

    return permission_checker


# User management functions
def create_user(user_create: UserCreate, permissions: List[str] = None) -> User:
    """Create a new user."""
    db = load_users_db()
    users = db.get("users", {})

    if user_create.username in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    if permissions is None:
        permissions = ["read", "predict"]  # Default permissions

    hashed_password = get_password_hash(user_create.password)

    user_data = {
        "username": user_create.username,
        "full_name": user_create.full_name,
        "email": user_create.email,
        "hashed_password": hashed_password,
        "disabled": False,
        "permissions": permissions
    }

    users[user_create.username] = user_data
    db["users"] = users
    save_users_db(db)

    return User(**{k: v for k, v in user_data.items() if k != "hashed_password"})


def change_password(username: str, current_password: str, new_password: str) -> bool:
    """Change a user's password."""
    user = get_user(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if not verify_password(current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )

    db = load_users_db()
    db["users"][username]["hashed_password"] = get_password_hash(new_password)
    save_users_db(db)

    return True


def update_user_permissions(username: str, permissions: List[str]) -> User:
    """Update a user's permissions (admin only)."""
    db = load_users_db()
    users = db.get("users", {})

    if username not in users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    users[username]["permissions"] = permissions
    db["users"] = users
    save_users_db(db)

    user_data = users[username]
    return User(**{k: v for k, v in user_data.items() if k != "hashed_password"})
