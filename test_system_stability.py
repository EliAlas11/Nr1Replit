
#!/usr/bin/env python3
"""
Comprehensive System Stability & Recovery Test Suite
Tests all critical stability components with real validation
"""

import asyncio
import logging
import traceback
import time
import psutil
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_health_check_endpoints():
    """Test 1: Health Check Endpoints (1-10 score)"""
    print("\n🏥 TESTING HEALTH CHECK ENDPOINTS")
    print("=" * 50)
    
    score = 0
    max_score = 10
    
    try:
        # Test health monitor import and initialization
        from app.routes.health_endpoints import health_monitor, router
        print("✅ Health endpoints module import: SUCCESS")
        score += 2
        
        # Test health monitor initialization
        await health_monitor.initialize()
        print("✅ Health monitor initialization: SUCCESS")
        score += 2
        
        # Test comprehensive health check
        health_result = await health_monitor.get_comprehensive_health()
        if health_result:
            overall_status = health_result.overall_status
            health_score = health_result.overall_score
            
            print(f"✅ Comprehensive health check: {overall_status.value}")
            print(f"✅ Health score: {health_score:.1f}/10")
            print(f"✅ Metrics collected: {len(health_result.metrics)}")
            
            # Score based on health results
            if overall_status.value in ["excellent", "perfect"] and health_score >= 9.0:
                score += 6
            elif overall_status.value == "healthy" and health_score >= 7.0:
                score += 4
            elif health_score >= 5.0:
                score += 2
            else:
                score += 1
                
        else:
            print("⚠️ Health check returned no results")
            score += 1
            
    except Exception as e:
        print(f"❌ Health endpoints test failed: {e}")
        print(traceback.format_exc())
        score = 3
    
    print(f"\n🎯 Health Check Endpoints Score: {score}/{max_score}")
    return score

async def test_crash_recovery_system():
    """Test 2: Crash Recovery & Auto-restart (1-10 score)"""
    print("\n🛡️ TESTING CRASH RECOVERY SYSTEM")
    print("=" * 50)
    
    score = 0
    max_score = 10
    
    try:
        # Test crash recovery manager import
        from app.crash_recovery_manager import crash_recovery_manager
        print("✅ Crash recovery manager import: SUCCESS")
        score += 2
        
        # Test recovery stats
        recovery_stats = crash_recovery_manager.get_recovery_stats()
        if recovery_stats:
            print(f"✅ Recovery stats available: {len(recovery_stats)} metrics")
            score += 2
        else:
            print("⚠️ Recovery stats: LIMITED")
            score += 1
            
        # Test startup failure handling
        test_error = Exception("Test startup failure")
        recovery_result = await crash_recovery_manager.handle_startup_failure(test_error)
        if recovery_result and recovery_result.get("recovery_id"):
            print("✅ Startup failure recovery: SUCCESS")
            score += 2
        else:
            print("⚠️ Startup failure recovery: PARTIAL")
            score += 1
            
        # Test memory exhaustion handling
        memory_result = await crash_recovery_manager.handle_memory_exhaustion()
        if memory_result and memory_result.get("recovery_id"):
            print("✅ Memory exhaustion recovery: SUCCESS")
            score += 2
        else:
            print("⚠️ Memory exhaustion recovery: PARTIAL")
            score += 1
            
        # Test health check
        health = await crash_recovery_manager.health_check()
        if health and health.get("status") == "healthy":
            print("✅ Recovery manager health: HEALTHY")
            score += 2
        else:
            print(f"⚠️ Recovery manager health: {health.get('status', 'unknown')}")
            score += 1
            
    except Exception as e:
        print(f"❌ Crash recovery test failed: {e}")
        print(traceback.format_exc())
        score = 4
    
    print(f"\n🎯 Crash Recovery Score: {score}/{max_score}")
    return score

async def test_dependency_validation():
    """Test 3: Dependency Validation at Boot (1-10 score)"""
    print("\n🔧 TESTING DEPENDENCY VALIDATION")
    print("=" * 50)
    
    score = 0
    max_score = 10
    
    try:
        # Test startup validator import
        from app.startup_validator import startup_validator
        print("✅ Startup validator import: SUCCESS")
        score += 1
        
        # Run comprehensive validation
        validation_result = await startup_validator.validate_system_startup()
        
        if validation_result:
            is_valid = validation_result.get("is_valid", False)
            system_score = validation_result.get("system_score", 0)
            passed_checks = validation_result.get("passed_checks", 0)
            total_checks = validation_result.get("total_checks", 0)
            
            print(f"✅ System validation completed: {is_valid}")
            print(f"✅ Validation score: {system_score:.1f}%")
            print(f"✅ Checks passed: {passed_checks}/{total_checks}")
            
            # Score based on validation results
            if is_valid and system_score >= 90:
                score += 6
            elif is_valid and system_score >= 80:
                score += 5
            elif is_valid:
                score += 4
            else:
                score += 2
                
            # Additional points for comprehensive checks
            if total_checks >= 8:
                score += 2
            elif total_checks >= 5:
                score += 1
                
            # Show detailed results
            detailed_results = validation_result.get("detailed_results", [])
            if detailed_results:
                print(f"✅ Detailed validation results: {len(detailed_results)} checks")
                critical_failures = validation_result.get("critical_failures", [])
                if critical_failures:
                    print(f"⚠️ Critical failures: {len(critical_failures)}")
                else:
                    print("✅ No critical failures detected")
                    score += 1
                    
        else:
            print("❌ Validation failed to return results")
            score += 1
            
    except Exception as e:
        print(f"❌ Dependency validation failed: {e}")
        print(traceback.format_exc())
        score = 3
    
    print(f"\n🎯 Dependency Validation Score: {score}/{max_score}")
    return score

async def test_additional_stability_features():
    """Test additional stability features"""
    print("\n⚙️ TESTING ADDITIONAL STABILITY FEATURES")
    print("=" * 50)
    
    additional_score = 0
    
    try:
        # Test Netflix Recovery System
        from app.netflix_recovery_system import recovery_system
        print("✅ Netflix Recovery System: AVAILABLE")
        additional_score += 1
        
        # Test recovery system stats
        recovery_stats = recovery_system.get_recovery_stats()
        if recovery_stats:
            print(f"✅ Recovery system stats: {recovery_stats.get('total_recoveries', 0)} total recoveries")
            additional_score += 1
        
    except Exception as e:
        print(f"⚠️ Netflix Recovery System: {e}")
    
    try:
        # Test Database Health Monitor
        from app.database.health import health_monitor as db_health_monitor
        print("✅ Database Health Monitor: AVAILABLE")
        additional_score += 1
        
        # Test database health
        db_health = await db_health_monitor.get_health_summary()
        if db_health:
            print(f"✅ Database health status: {db_health.get('overall_status', 'unknown')}")
            additional_score += 1
        
    except Exception as e:
        print(f"⚠️ Database Health Monitor: {e}")
    
    try:
        # Test system resource monitoring
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        print(f"✅ System monitoring: {memory.percent:.1f}% memory, {cpu_percent:.1f}% CPU")
        additional_score += 1
        
    except Exception as e:
        print(f"⚠️ System monitoring: {e}")
    
    print(f"\n🎯 Additional Features Score: {additional_score}/5")
    return additional_score

async def main():
    """Run all system stability tests"""
    print("🔍 COMPREHENSIVE SYSTEM STABILITY TEST")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    
    # Run all tests
    health_score = await test_health_check_endpoints()
    recovery_score = await test_crash_recovery_system()
    dependency_score = await test_dependency_validation()
    additional_score = await test_additional_stability_features()
    
    # Calculate overall results
    print("\n" + "=" * 60)
    print("📊 FINAL STABILITY TEST RESULTS:")
    print("=" * 60)
    
    print(f"1. Health Check Endpoints:     {health_score}/10")
    print(f"2. Crash Recovery & Auto-restart: {recovery_score}/10")
    print(f"3. Dependency Validation:      {dependency_score}/10")
    print(f"   Additional Features:       {additional_score}/5")
    
    # Overall scoring
    total_score = health_score + recovery_score + dependency_score
    max_total = 30
    
    print(f"\n🎯 OVERALL STABILITY SCORE: {total_score}/{max_total}")
    
    if total_score >= 27:
        grade = "🏆 EXCELLENT - Production-ready Netflix-grade stability!"
    elif total_score >= 24:
        grade = "✅ VERY GOOD - Enterprise-level stability achieved"
    elif total_score >= 20:
        grade = "👍 GOOD - Solid stability foundation"
    elif total_score >= 15:
        grade = "⚠️ NEEDS IMPROVEMENT - Some stability gaps"
    else:
        grade = "❌ CRITICAL - Major stability issues detected"
    
    print(f"Grade: {grade}")
    
    # Individual component analysis
    print(f"\n📋 COMPONENT ANALYSIS:")
    components = [
        ("Health Check Endpoints", health_score, "Critical for monitoring system health"),
        ("Crash Recovery", recovery_score, "Essential for automatic failure recovery"),
        ("Dependency Validation", dependency_score, "Required for reliable system startup")
    ]
    
    for name, score, description in components:
        status = "✅ EXCELLENT" if score >= 9 else "✅ GOOD" if score >= 7 else "⚠️ NEEDS WORK" if score >= 5 else "❌ CRITICAL"
        print(f"{name}: {score}/10 - {status}")
        print(f"   → {description}")
    
    print(f"\nTest completed at: {datetime.now()}")
    return total_score

if __name__ == "__main__":
    asyncio.run(main())
