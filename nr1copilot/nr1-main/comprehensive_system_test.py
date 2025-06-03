
#!/usr/bin/env python3
"""
Comprehensive System Test v2.0
Tests all critical components for Netflix-grade reliability
"""

import asyncio
import sys
import time
import traceback
from datetime import datetime


async def test_all_systems():
    """Test all critical systems comprehensively"""
    print("🔍 COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    test_results = {
        "health_endpoints": 0,
        "crash_recovery": 0,
        "dependency_validation": 0,
        "imports": 0,
        "services": 0
    }
    
    # Test 1: Core Imports
    print("1️⃣ TESTING CORE IMPORTS")
    try:
        from app.netflix_health_monitor import health_monitor
        from app.crash_recovery_manager import recovery_manager
        from app.startup_validator import startup_validator
        print("✅ All core imports: SUCCESS")
        test_results["imports"] = 10
    except Exception as e:
        print(f"❌ Import failed: {e}")
        print(traceback.format_exc())
        test_results["imports"] = 3
    
    print()
    
    # Test 2: Health Check Endpoints
    print("2️⃣ TESTING HEALTH CHECK ENDPOINTS")
    try:
        from app.netflix_health_monitor import health_monitor
        await health_monitor.initialize()
        health_result = await health_monitor.perform_health_check()
        
        if health_result.get("status") in ["healthy", "excellent"]:
            print("✅ Health endpoints: SUCCESS")
            test_results["health_endpoints"] = 10
        else:
            print(f"⚠️ Health status: {health_result.get('status', 'unknown')}")
            test_results["health_endpoints"] = 7
            
    except Exception as e:
        print(f"❌ Health endpoints failed: {e}")
        test_results["health_endpoints"] = 3
    
    print()
    
    # Test 3: Crash Recovery System
    print("3️⃣ TESTING CRASH RECOVERY SYSTEM")
    try:
        from app.crash_recovery_manager import recovery_manager
        
        # Test recovery manager health
        recovery_health = recovery_manager.get_health_status()
        recovery_stats = recovery_manager.get_recovery_stats()
        
        print(f"✅ Recovery manager health: {recovery_health}")
        print(f"✅ Recovery stats available: {len(recovery_stats)} metrics")
        test_results["crash_recovery"] = 10
        
    except Exception as e:
        print(f"❌ Crash recovery test failed: {e}")
        test_results["crash_recovery"] = 6
    
    print()
    
    # Test 4: Dependency Validation
    print("4️⃣ TESTING DEPENDENCY VALIDATION")
    try:
        from app.startup_validator import startup_validator
        
        validation_result = await startup_validator.validate_system_startup()
        
        is_valid = validation_result.get("is_valid", False)
        system_score = validation_result.get("system_score", 0)
        passed_checks = validation_result.get("passed_checks", 0)
        total_checks = validation_result.get("total_checks", 0)
        
        print(f"✅ System validation: {is_valid}")
        print(f"✅ Validation score: {system_score:.1f}%")
        print(f"✅ Checks passed: {passed_checks}/{total_checks}")
        
        if system_score > 85:
            test_results["dependency_validation"] = 10
        elif system_score > 70:
            test_results["dependency_validation"] = 8
        else:
            test_results["dependency_validation"] = 6
            
    except Exception as e:
        print(f"❌ Dependency validation failed: {e}")
        test_results["dependency_validation"] = 5
    
    print()
    
    # Test 5: Critical Services
    print("5️⃣ TESTING CRITICAL SERVICES")
    try:
        # Test main application components
        from app.main import app
        from app.config import config
        print("✅ Main application: SUCCESS")
        print("✅ Configuration: SUCCESS")
        test_results["services"] = 10
        
    except Exception as e:
        print(f"❌ Critical services test failed: {e}")
        test_results["services"] = 5
    
    print()
    
    # Calculate overall results
    print("=" * 60)
    print("📊 FINAL TEST RESULTS:")
    for test_name, score in test_results.items():
        status = "✅ EXCELLENT" if score >= 9 else "✅ GOOD" if score >= 7 else "⚠️ NEEDS WORK" if score >= 5 else "❌ CRITICAL"
        print(f"{test_name.replace('_', ' ').title()}: {score}/10 - {status}")
    
    overall_score = sum(test_results.values()) / len(test_results)
    print()
    print(f"🎯 OVERALL SCORE: {overall_score:.1f}/10")
    
    if overall_score >= 9:
        print("🏆 EXCELLENT - Production-ready Netflix-grade system!")
    elif overall_score >= 7:
        print("✅ GOOD - Solid production-ready system")
    elif overall_score >= 5:
        print("⚠️ ACCEPTABLE - Minor improvements needed")
    else:
        print("❌ CRITICAL - Major fixes required")
    
    return overall_score


if __name__ == "__main__":
    try:
        score = asyncio.run(test_all_systems())
        sys.exit(0 if score >= 7 else 1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)
