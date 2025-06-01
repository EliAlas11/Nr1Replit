
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
"""
Security Utilities
Netflix-level security implementation
"""

import hashlib
import secrets
from typing import Dict, Any, Optional

class SecurityManager:
    """Netflix-level security manager"""
    
    def __init__(self):
        self.failed_attempts = {}
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return salt + password_hash.hex()
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        salt = stored_hash[:64]
        stored_password_hash = stored_hash[64:]
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return password_hash.hex() == stored_password_hash
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)
    
    def validate_input(self, data: str, max_length: int = 1000) -> bool:
        """Basic input validation"""
        if not data or len(data) > max_length:
            return False
        
        # Check for common injection patterns
        dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
        data_lower = data.lower()
        
        return not any(pattern in data_lower for pattern in dangerous_patterns)
    
    def check_rate_limit_security(self, client_ip: str) -> bool:
        """Check if client IP is rate limited for security"""
        if client_ip in self.failed_attempts:
            attempts = self.failed_attempts[client_ip]
            if attempts.get('count', 0) > 10:  # Max 10 failed attempts
                return False
        
        return True
    
    def record_failed_attempt(self, client_ip: str):
        """Record failed authentication attempt"""
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = {'count': 0}
        
        self.failed_attempts[client_ip]['count'] += 1
