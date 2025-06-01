
"""
ViralClip Pro - Netflix-Level Security Management
Advanced security features with threat detection and prevention
"""

import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import ipaddress
import re
import json

try:
    import jwt
    HAS_JWT = True
except ImportError:
    HAS_JWT = False

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class SecurityManager:
    """Netflix-level security management"""
    
    def __init__(self):
        self.failed_attempts = {}  # IP -> {count, last_attempt, blocked_until}
        self.suspicious_patterns = {
            'sql_injection': [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bINSERT\b.*\bINTO\b)",
                r"(\bDROP\b.*\bTABLE\b)",
                r"(\bDELETE\b.*\bFROM\b)"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"/etc/passwd",
                r"/proc/",
                r"\\windows\\"
            ]
        }
        self.rate_limits = {}  # endpoint -> {requests: [], limit, window}
        
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt using secure algorithm"""
        if not salt:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 with SHA-256
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
        )
        
        return pwdhash.hex(), salt
    
    def verify_password(self, password: str, hash_hex: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            expected_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return hmac.compare_digest(expected_hash.hex(), hash_hex)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def create_jwt_token(
        self, 
        payload: Dict[str, Any], 
        expires_in: int = 3600,
        secret_key: Optional[str] = None
    ) -> Optional[str]:
        """Create JWT token with expiration"""
        if not HAS_JWT:
            logger.warning("JWT library not available")
            return None
        
        try:
            secret = secret_key or settings.secret_key
            now = datetime.utcnow()
            
            token_payload = {
                **payload,
                'iat': now,
                'exp': now + timedelta(seconds=expires_in),
                'jti': self.generate_secure_token(16)  # JWT ID for revocation
            }
            
            return jwt.encode(token_payload, secret, algorithm='HS256')
            
        except Exception as e:
            logger.error(f"JWT creation error: {e}")
            return None
    
    def verify_jwt_token(
        self, 
        token: str, 
        secret_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        if not HAS_JWT:
            logger.warning("JWT library not available")
            return None
        
        try:
            secret = secret_key or settings.secret_key
            payload = jwt.decode(token, secret, algorithms=['HS256'])
            
            # Additional validation
            if 'exp' in payload and datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                logger.warning("Token expired")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT verification error: {e}")
            return None
    
    def validate_request_signature(
        self, 
        payload: str, 
        signature: str, 
        secret: str,
        timestamp: Optional[str] = None,
        tolerance: int = 300
    ) -> bool:
        """Validate request signature (webhook style)"""
        try:
            # Check timestamp if provided (prevent replay attacks)
            if timestamp:
                try:
                    ts = int(timestamp)
                    if abs(time.time() - ts) > tolerance:
                        logger.warning("Request timestamp outside tolerance")
                        return False
                    payload_to_sign = f"{timestamp}.{payload}"
                except (ValueError, TypeError):
                    logger.warning("Invalid timestamp format")
                    return False
            else:
                payload_to_sign = payload
            
            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload_to_sign.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(f"sha256={expected_signature}", signature)
            
        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False
    
    def detect_suspicious_content(self, content: str) -> Dict[str, Any]:
        """Detect suspicious patterns in content"""
        threats = []
        
        for threat_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    threats.append({
                        'type': threat_type,
                        'pattern': pattern,
                        'severity': 'high' if threat_type in ['sql_injection', 'xss'] else 'medium'
                    })
        
        return {
            'suspicious': len(threats) > 0,
            'threats': threats,
            'risk_score': min(len(threats) * 25, 100)
        }
    
    def validate_ip_address(self, ip_address: str) -> Dict[str, Any]:
        """Validate and analyze IP address"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            analysis = {
                'valid': True,
                'version': ip.version,
                'is_private': ip.is_private,
                'is_loopback': ip.is_loopback,
                'is_multicast': ip.is_multicast,
                'is_reserved': ip.is_reserved,
                'risk_level': 'low'
            }
            
            # Risk assessment
            if ip.is_private or ip.is_loopback:
                analysis['risk_level'] = 'very_low'
            elif ip.is_reserved or ip.is_multicast:
                analysis['risk_level'] = 'medium'
            
            # Check against known bad ranges (simplified)
            if self._is_known_bad_ip(str(ip)):
                analysis['risk_level'] = 'high'
                analysis['blocked'] = True
            
            return analysis
            
        except ValueError:
            return {
                'valid': False,
                'error': 'Invalid IP address format',
                'risk_level': 'high'
            }
    
    def _is_known_bad_ip(self, ip: str) -> bool:
        """Check if IP is in known bad ranges"""
        # Simplified - in production, this would check against threat intelligence
        bad_ranges = [
            '10.0.0.0/8',    # Example bad range
            '192.168.0.0/16'  # Example (normally private, but for demo)
        ]
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            for bad_range in bad_ranges:
                if ip_obj in ipaddress.ip_network(bad_range):
                    return True
        except ValueError:
            pass
        
        return False
    
    def track_failed_login(self, ip_address: str, username: str = None) -> Dict[str, Any]:
        """Track failed login attempts and implement progressive delays"""
        now = datetime.utcnow()
        
        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = {
                'count': 0,
                'last_attempt': now,
                'blocked_until': None,
                'usernames': set()
            }
        
        attempt_info = self.failed_attempts[ip_address]
        attempt_info['count'] += 1
        attempt_info['last_attempt'] = now
        
        if username:
            attempt_info['usernames'].add(username)
        
        # Progressive blocking
        if attempt_info['count'] >= 5:
            # Block for increasing duration
            block_duration = min(2 ** (attempt_info['count'] - 5), 3600)  # Max 1 hour
            attempt_info['blocked_until'] = now + timedelta(seconds=block_duration)
            
            logger.warning(
                f"IP {ip_address} blocked for {block_duration}s after {attempt_info['count']} failed attempts"
            )
        
        return {
            'attempts': attempt_info['count'],
            'blocked': attempt_info['blocked_until'] is not None and now < attempt_info['blocked_until'],
            'blocked_until': attempt_info['blocked_until'],
            'unique_usernames': len(attempt_info['usernames'])
        }
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is currently blocked"""
        if ip_address not in self.failed_attempts:
            return False
        
        attempt_info = self.failed_attempts[ip_address]
        blocked_until = attempt_info.get('blocked_until')
        
        if blocked_until and datetime.utcnow() < blocked_until:
            return True
        
        # Clean up expired blocks
        if blocked_until and datetime.utcnow() >= blocked_until:
            attempt_info['blocked_until'] = None
        
        return False
    
    def reset_failed_attempts(self, ip_address: str):
        """Reset failed attempts for successful login"""
        if ip_address in self.failed_attempts:
            del self.failed_attempts[ip_address]
    
    def check_rate_limit(
        self, 
        endpoint: str, 
        identifier: str, 
        limit: int = 100, 
        window: int = 3600
    ) -> Dict[str, Any]:
        """Check rate limit for endpoint and identifier"""
        key = f"{endpoint}:{identifier}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                'requests': [],
                'limit': limit,
                'window': window
            }
        
        rate_info = self.rate_limits[key]
        
        # Clean old requests outside window
        rate_info['requests'] = [
            req_time for req_time in rate_info['requests'] 
            if now - req_time < window
        ]
        
        # Check if limit exceeded
        if len(rate_info['requests']) >= limit:
            return {
                'allowed': False,
                'limit': limit,
                'remaining': 0,
                'reset_time': min(rate_info['requests']) + window,
                'retry_after': min(rate_info['requests']) + window - now
            }
        
        # Add current request
        rate_info['requests'].append(now)
        
        return {
            'allowed': True,
            'limit': limit,
            'remaining': limit - len(rate_info['requests']),
            'reset_time': now + window,
            'retry_after': 0
        }
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', '', filename)
        sanitized = re.sub(r'[\\/]', '_', sanitized)
        sanitized = re.sub(r'\.\.+', '.', sanitized)
        
        # Ensure filename is not empty and not too long
        if not sanitized or sanitized.strip() == '':
            sanitized = f"file_{int(time.time())}"
        
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:250] + ext
        
        return sanitized
    
    def validate_file_upload(
        self, 
        filename: str, 
        content_type: str, 
        file_size: int,
        allowed_types: List[str] = None,
        max_size: int = None
    ) -> Dict[str, Any]:
        """Comprehensive file upload validation"""
        
        allowed_types = allowed_types or settings.allowed_video_formats
        max_size = max_size or settings.max_file_size
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_filename': self.sanitize_filename(filename)
        }
        
        # File size check
        if file_size > max_size:
            validation['valid'] = False
            validation['errors'].append(f"File size {file_size} exceeds limit {max_size}")
        
        # File type check
        file_ext = os.path.splitext(filename)[1].lower().lstrip('.')
        if file_ext not in allowed_types:
            validation['valid'] = False
            validation['errors'].append(f"File type '{file_ext}' not allowed")
        
        # Content type validation
        expected_content_types = {
            'mp4': 'video/mp4',
            'mov': 'video/quicktime',
            'avi': 'video/x-msvideo',
            'mkv': 'video/x-matroska',
            'webm': 'video/webm'
        }
        
        expected_type = expected_content_types.get(file_ext)
        if expected_type and content_type != expected_type:
            validation['warnings'].append(
                f"Content type '{content_type}' doesn't match extension '{file_ext}'"
            )
        
        # Filename security check
        suspicious_content = self.detect_suspicious_content(filename)
        if suspicious_content['suspicious']:
            validation['valid'] = False
            validation['errors'].append("Filename contains suspicious content")
        
        return validation
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        timestamp = str(int(time.time()))
        payload = f"{session_id}:{timestamp}"
        
        signature = hmac.new(
            settings.secret_key.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"{timestamp}.{signature}"
    
    def validate_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token"""
        try:
            timestamp_str, signature = token.split('.', 1)
            timestamp = int(timestamp_str)
            
            # Check age
            if time.time() - timestamp > max_age:
                return False
            
            # Verify signature
            payload = f"{session_id}:{timestamp_str}"
            expected_signature = hmac.new(
                settings.secret_key.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, TypeError):
            return False
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' wss: ws:; "
                "media-src 'self' blob:; "
                "object-src 'none'; "
                "base-uri 'self'"
            ),
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': (
                'camera=(), microphone=(), geolocation=(), '
                'payment=(), usb=(), magnetometer=(), gyroscope=()'
            )
        }
    
    def audit_log(
        self, 
        action: str, 
        user_id: str = None, 
        ip_address: str = None,
        details: Dict[str, Any] = None
    ):
        """Log security-relevant actions"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details or {},
            'session_id': self.generate_secure_token(16)
        }
        
        # Log to security log
        security_logger = logging.getLogger('security')
        security_logger.info(json.dumps(log_entry))
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary"""
        now = datetime.utcnow()
        
        # Count blocked IPs
        blocked_ips = sum(
            1 for attempt_info in self.failed_attempts.values()
            if attempt_info.get('blocked_until') and now < attempt_info['blocked_until']
        )
        
        # Count recent failed attempts
        recent_failures = sum(
            1 for attempt_info in self.failed_attempts.values()
            if (now - attempt_info['last_attempt']).total_seconds() < 3600
        )
        
        return {
            'blocked_ips': blocked_ips,
            'recent_failed_attempts': recent_failures,
            'total_tracked_ips': len(self.failed_attempts),
            'rate_limited_endpoints': len(self.rate_limits),
            'jwt_available': HAS_JWT,
            'security_features_enabled': [
                'rate_limiting',
                'failed_attempt_tracking',
                'suspicious_content_detection',
                'file_upload_validation',
                'csrf_protection',
                'security_headers'
            ]
        }
