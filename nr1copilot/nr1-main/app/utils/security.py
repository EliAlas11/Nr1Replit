
"""
Netflix-Level Security Management
"""

import hashlib
import secrets
import time
from typing import Dict, Any, Optional
import jwt
from passlib.context import CryptContext

class SecurityManager:
    """Netflix-level security utilities"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = "your-secret-key-here"  # Should be from environment
        self.algorithm = "HS256"
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[int] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = time.time() + (expires_delta or 3600)  # Default 1 hour
        to_encode.update({"exp": expire})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            return None
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)
    
    def hash_file(self, file_path: str) -> str:
        """Generate file hash for integrity checking"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except FileNotFoundError:
            return ""
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for security"""
        import re
        # Remove any path traversal attempts
        filename = filename.replace("..", "").replace("/", "").replace("\\", "")
        # Only allow alphanumeric, dots, dashes, and underscores
        filename = re.sub(r'[^a-zA-Z0-9.\-_]', '', filename)
        return filename[:255]  # Limit length
