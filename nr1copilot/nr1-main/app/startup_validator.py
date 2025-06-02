
"""
Netflix-Grade Startup Validator
Comprehensive system validation before application startup
"""

import sys
import logging
import importlib
import traceback
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class StartupValidator:
    """Netflix-tier startup validation system"""
    
    def __init__(self):
        self.critical_modules = [
            "fastapi",
            "uvicorn", 
            "pydantic_settings",
            "psutil",
            "websockets",
            "starlette",
            "typing"
        ]
        
        self.critical_app_modules = [
            "app.config",
            "app.middleware.security",
            "app.middleware.performance", 
            "app.middleware.error_handler"
        ]
        
        self.validation_results: Dict[str, Any] = {}

    def validate_critical_imports(self) -> bool:
        """Validate all critical imports"""
        logger.info("🔍 Validating critical imports...")
        
        failed_imports = []
        
        # Check external dependencies
        for module in self.critical_modules:
            try:
                importlib.import_module(module)
                logger.debug(f"✅ {module} - OK")
            except ImportError as e:
                failed_imports.append((module, str(e)))
                logger.error(f"❌ {module} - FAILED: {e}")
        
        # Check internal app modules
        for module in self.critical_app_modules:
            try:
                importlib.import_module(module)
                logger.debug(f"✅ {module} - OK")
            except ImportError as e:
                failed_imports.append((module, str(e)))
                logger.error(f"❌ {module} - FAILED: {e}")
        
        self.validation_results["failed_imports"] = failed_imports
        
        if failed_imports:
            logger.critical("💥 Critical import validation FAILED")
            for module, error in failed_imports:
                logger.critical(f"   - {module}: {error}")
            return False
        
        logger.info("✅ All critical imports validated successfully")
        return True

    def validate_environment(self) -> bool:
        """Validate environment configuration"""
        logger.info("🔍 Validating environment...")
        
        try:
            from app.config import settings
            logger.info(f"✅ Environment: {settings.environment}")
            logger.info(f"✅ Debug mode: {settings.debug}")
            return True
        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
            return False

    def validate_typing_support(self) -> bool:
        """Validate typing support"""
        logger.info("🔍 Validating typing support...")
        
        try:
            from typing import Any, Dict, List, Optional, Union
            logger.debug("✅ Typing module imports - OK")
            return True
        except ImportError as e:
            logger.error(f"❌ Typing validation failed: {e}")
            return False

    def run_full_validation(self) -> bool:
        """Run complete startup validation"""
        logger.info("🚀 Starting Netflix-grade startup validation...")
        
        validations = [
            ("Critical Imports", self.validate_critical_imports),
            ("Environment", self.validate_environment),
            ("Typing Support", self.validate_typing_support)
        ]
        
        failed_validations = []
        
        for name, validator in validations:
            try:
                if not validator():
                    failed_validations.append(name)
            except Exception as e:
                logger.error(f"💥 Validation '{name}' crashed: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                failed_validations.append(name)
        
        if failed_validations:
            logger.critical("💥 STARTUP VALIDATION FAILED")
            logger.critical(f"Failed validations: {', '.join(failed_validations)}")
            return False
        
        logger.info("✅ All startup validations passed - Netflix-grade quality achieved")
        return True

def validate_startup() -> bool:
    """Main startup validation entry point"""
    validator = StartupValidator()
    return validator.run_full_validation()

if __name__ == "__main__":
    if not validate_startup():
        sys.exit(1)
    print("✅ Startup validation completed successfully")
