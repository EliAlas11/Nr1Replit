
#!/usr/bin/env python3
"""
Netflix-Grade Import Validation System
Comprehensive testing of all application imports and dependencies
"""

import sys
import traceback
import asyncio
import logging
from datetime import datetime

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)

class NetflixGradeImportValidator:
    """Netflix-tier import validation with comprehensive error reporting"""
    
    def __init__(self):
        self.results = {}
        self.critical_errors = []
        self.warnings = []
        
    def test_import(self, module_name: str, import_statement: str) -> bool:
        """Test individual import with detailed error reporting"""
        try:
            exec(import_statement)
            logger.info(f"✅ {module_name} import: SUCCESS")
            self.results[module_name] = {"status": "SUCCESS", "error": None}
            return True
        except Exception as e:
            logger.error(f"❌ {module_name} import: FAILED - {e}")
            self.results[module_name] = {"status": "FAILED", "error": str(e)}
            if "critical" in module_name.lower():
                self.critical_errors.append(f"{module_name}: {e}")
            else:
                self.warnings.append(f"{module_name}: {e}")
            return False

    def test_core_dependencies(self):
        """Test core Python and external dependencies"""
        print("🔍 Testing core dependencies...")
        
        dependencies = [
            ("FastAPI", "import fastapi"),
            ("Uvicorn", "import uvicorn"),
            ("Pydantic", "import pydantic"),
            ("Pydantic Settings", "from pydantic_settings import BaseSettings"),
            ("PSUtil", "import psutil"),
            ("WebSockets", "import websockets"),
            ("Python Multipart", "import multipart"),
            ("AsyncIO", "import asyncio"),
            ("Collections", "from collections import deque"),
            ("Statistics", "import statistics"),
            ("JSON", "import json"),
            ("Time", "import time"),
            ("Threading", "import threading"),
            ("Logging", "import logging"),
            ("Datetime", "from datetime import datetime, timedelta"),
            ("Typing", "from typing import Dict, List, Optional, Any"),
            ("Dataclasses", "from dataclasses import dataclass, field")
        ]
        
        success_count = 0
        for name, import_stmt in dependencies:
            if self.test_import(name, import_stmt):
                success_count += 1
                
        print(f"📊 Core dependencies: {success_count}/{len(dependencies)} successful")
        return success_count == len(dependencies)

    def test_application_modules(self):
        """Test application-specific modules"""
        print("🔍 Testing application modules...")
        
        modules = [
            ("App Config", "from app.config import get_settings"),
            ("App Utils Package", "import app.utils"),
            ("Cache Manager", "from app.utils import cache"),
            ("Health Monitor", "from app.utils import HealthMonitor"),
            ("Metrics Collector", "from app.utils import MetricsCollector"),
            ("Performance Monitor", "from app.utils import PerformanceMonitor"),
            ("Middleware Package", "import app.middleware"),
            ("Performance Middleware", "from app.middleware.performance import PerformanceMiddleware"),
            ("Security Middleware", "from app.middleware.security import SecurityMiddleware"),
            ("Error Handler", "from app.middleware.error_handler import ErrorHandlerMiddleware")
        ]
        
        success_count = 0
        for name, import_stmt in modules:
            if self.test_import(name, import_stmt):
                success_count += 1
                
        print(f"📊 Application modules: {success_count}/{len(modules)} successful")
        return success_count == len(modules)

    def test_main_application(self):
        """Test main application import"""
        print("🔍 Testing main application...")
        
        try:
            from app.main import app
            logger.info("✅ Main app import: SUCCESS")
            self.results["Main Application"] = {"status": "SUCCESS", "error": None}
            return True
        except Exception as e:
            logger.error(f"❌ Main app import: FAILED - {e}")
            traceback.print_exc()
            self.results["Main Application"] = {"status": "FAILED", "error": str(e)}
            self.critical_errors.append(f"Main Application: {e}")
            return False

    async def test_async_functionality(self):
        """Test async functionality"""
        print("🔍 Testing async functionality...")
        
        try:
            # Test basic async functionality
            await asyncio.sleep(0.1)
            
            # Test application state initialization
            from app.main import app_state
            await app_state.initialize()
            
            logger.info("✅ Async functionality: SUCCESS")
            self.results["Async Functionality"] = {"status": "SUCCESS", "error": None}
            return True
        except Exception as e:
            logger.error(f"❌ Async functionality: FAILED - {e}")
            self.results["Async Functionality"] = {"status": "FAILED", "error": str(e)}
            return False

    async def perform_complete_validation(self):
        """Perform complete Netflix-grade validation"""
        print("🚀 Starting Netflix-Grade Import Validation")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Test core dependencies
        core_success = self.test_core_dependencies()
        print()
        
        # Test application modules
        app_success = self.test_application_modules()
        print()
        
        # Test main application
        main_success = self.test_main_application()
        print()
        
        # Test async functionality
        async_success = await self.test_async_functionality()
        print()
        
        # Generate final report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("=" * 60)
        print("📋 NETFLIX-GRADE VALIDATION REPORT")
        print("=" * 60)
        
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results.values() if r["status"] == "SUCCESS"])
        
        print(f"🕒 Duration: {duration:.2f} seconds")
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {total_tests - successful_tests}")
        print(f"📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if self.critical_errors:
            print(f"\n🚨 CRITICAL ERRORS ({len(self.critical_errors)}):")
            for error in self.critical_errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Overall status
        if successful_tests == total_tests:
            print("\n🎯 VALIDATION STATUS: ✅ PASSED - Netflix-Grade Quality Achieved!")
            validation_status = "PASSED"
        elif len(self.critical_errors) == 0:
            print("\n🎯 VALIDATION STATUS: ⚠️  PASSED WITH WARNINGS - Minor issues detected")
            validation_status = "PASSED_WITH_WARNINGS"
        else:
            print("\n🎯 VALIDATION STATUS: ❌ FAILED - Critical issues require attention")
            validation_status = "FAILED"
        
        print("=" * 60)
        
        return {
            "validation_status": validation_status,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": (successful_tests/total_tests)*100,
            "duration_seconds": duration,
            "critical_errors": self.critical_errors,
            "warnings": self.warnings,
            "detailed_results": self.results,
            "netflix_grade": successful_tests == total_tests,
            "timestamp": datetime.now().isoformat()
        }

async def main():
    """Main validation entry point"""
    validator = NetflixGradeImportValidator()
    result = await validator.perform_complete_validation()
    
    # Exit with appropriate code
    if result["validation_status"] == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
