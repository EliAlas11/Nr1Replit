
#!/usr/bin/env python3
"""
Netflix-Grade Import Validation Test
Verifies all critical imports work correctly before startup
"""

import sys
import traceback

def test_critical_imports():
    """Test all critical imports that cause startup failures"""
    print("🔍 Testing Netflix-grade import validation...")
    
    failed_imports = []
    
    # Test core FastAPI imports
    try:
        from fastapi import FastAPI
        print("✅ FastAPI import: SUCCESS")
    except Exception as e:
        failed_imports.append(f"FastAPI: {e}")
        print(f"❌ FastAPI import: FAILED - {e}")
    
    # Test application config
    try:
        from app.config import get_settings
        settings = get_settings()
        print("✅ App config import: SUCCESS")
    except Exception as e:
        failed_imports.append(f"App config: {e}")
        print(f"❌ App config import: FAILED - {e}")
    
    # Test utils package
    try:
        from app.utils import cache, MetricsCollector, PerformanceMonitor, HealthMonitor
        print("✅ Utils package import: SUCCESS")
    except Exception as e:
        failed_imports.append(f"Utils package: {e}")
        print(f"❌ Utils package import: FAILED - {e}")
        traceback.print_exc()
    
    # Test middleware
    try:
        from app.middleware.security import SecurityMiddleware
        from app.middleware.performance import PerformanceMiddleware
        from app.middleware.error_handler import ErrorHandlerMiddleware
        print("✅ Middleware imports: SUCCESS")
    except Exception as e:
        failed_imports.append(f"Middleware: {e}")
        print(f"❌ Middleware imports: FAILED - {e}")
    
    # Test main app import
    try:
        from app.main import app
        print("✅ Main app import: SUCCESS")
    except Exception as e:
        failed_imports.append(f"Main app: {e}")
        print(f"❌ Main app import: FAILED - {e}")
        traceback.print_exc()
    
    if failed_imports:
        print(f"\n❌ Import validation FAILED with {len(failed_imports)} errors:")
        for error in failed_imports:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All Netflix-grade imports validated successfully!")
        return True

if __name__ == "__main__":
    success = test_critical_imports()
    sys.exit(0 if success else 1)
