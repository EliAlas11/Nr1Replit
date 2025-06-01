"""
S3 Service Layer (Refactored)

- Handles file upload logic to AWS S3.
- All business logic, validation, and error handling for S3 operations is centralized here.
- Designed for auditability, security, and testability (Stripe/Netflix standards).
- All functions are stateless and side-effect free except for file I/O.
- Implements audit logging and custom exceptions for compliance.
"""

import logging
from typing import Dict, Any
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from ..config import settings

logger = logging.getLogger("s3_service")

class S3ServiceError(Exception):
    """Custom exception for S3Service errors."""
    pass

s3 = boto3.client(
    's3',
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
)

def upload_file_to_s3(file_path: str, key: str) -> Dict[str, Any]:
    """
    Upload a file to AWS S3.

    Args:
        file_path (str): Local path to the file.
        key (str): S3 object key.

    Returns:
        dict[str, Any]: On success: {"success": True, "key": str}.

    Raises:
        S3ServiceError: If upload fails.
    """
    try:
        logger.info(f"Uploading file to S3: {file_path} -> {key}")
        s3.upload_file(file_path, settings.AWS_S3_BUCKET, key, ExtraArgs={"ACL": "public-read"})
        logger.info(f"File uploaded to S3: {key}")
        return {"success": True, "key": key}
    except (BotoCoreError, ClientError, Exception) as e:
        logger.error(f"S3 upload error: {e}")
        raise S3ServiceError(f"S3 upload error: {e}")

# Test block for service sanity (not for production)
if __name__ == "__main__":
    import sys
    try:
        # This is a stub test; replace with real file paths for actual testing
        print(upload_file_to_s3("test.txt", "test/test.txt"))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
