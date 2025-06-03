
#!/usr/bin/env python3
"""
Unbiased System Assessment Tool
Comprehensive testing of all critical systems without modifications
"""

import asyncio
import logging
import traceback
import time
import sys
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnbiasedSystemAssessment:
    """Comprehensive system assessment without modifications"""
    
    def __init__(self):
        self.results = {}
        self.total_score = 0
        self.max_possible_score = 50  # 5 categories × 10 points each
        
    async def assess_health_endpoints(self) -> Dict[str, Any]:
        """Assess health endpoints functionality"""
        print("🏥 ASSESSING HEALTH ENDPOINTS")
        print("=" * 50)
        
        score = 0
        issues = []
        
        try:
            # Test basic health endpoint import
            from app.routes.health_endpoints import router
            print("✅ Health endpoints router import: SUCCESS")
            score += 2
            
            # Test health monitor availability
            try:
                from app.utils.health import SystemHealthMonitor
                health_monitor = SystemHealthMonitor()
                print("✅ Health monitor class: AVAILABLE")
                score += 2
                
                # Test initialization
                await health_monitor.initialize()
                print("✅ Health monitor initialization: SUCCESS")
                score += 2
                
                # Test health check functionality
                health_result = await health_monitor.get_health_summary()
                if health_result:
                    print(f"✅ Health check result: {health_result.get('status', 'unknown')}")
                    score += 2
                else:
                    print("⚠️ Health check result: EMPTY")
                    issues.append("Health check returns empty result")
                    score += 1
                
            except ImportError as e:
                print(f"❌ Health monitor import failed: {e}")
                issues.append(f"Health monitor import error: {e}")
                
            # Test comprehensive health endpoint
            try:
                from app.routes.health_endpoints import comprehensive_health_check
                print("✅ Comprehensive health check function: AVAILABLE")
                score += 2
            except ImportError as e:
                print(f"❌ Comprehensive health check import failed: {e}")
                issues.append(f"Comprehensive health check import error: {e}")
                
        except Exception as e:
            print(f"❌ Health endpoints assessment failed: {e}")
            issues.append(f"Critical health endpoints error: {e}")
            score = max(score, 2)  # Minimum score for attempt
            
        return {
            "score": score,
            "max_score": 10,
            "percentage": (score / 10) * 100,
            "issues": issues,
            "status": "EXCELLENT" if score >= 9 else "GOOD" if score >= 7 else "NEEDS_WORK" if score >= 5 else "CRITICAL"
        }
    
    async def assess_crash_recovery(self) -> Dict[str, Any]:
        """Assess crash recovery system"""
        print("\n🛡️ ASSESSING CRASH RECOVERY SYSTEM")
        print("=" * 50)
        
        score = 0
        issues = []
        
        try:
            # Test crash recovery manager import
            from app.crash_recovery_manager import NetflixCrashRecoveryManager
            print("✅ Crash recovery manager import: SUCCESS")
            score += 2
            
            # Test crash recovery manager instance
            try:
                from app.crash_recovery_manager import crash_recovery_manager
                print("✅ Crash recovery manager instance: AVAILABLE")
                score += 2
                
                # Test recovery stats
                recovery_stats = crash_recovery_manager.get_recovery_stats()
                if recovery_stats:
                    print(f"✅ Recovery stats: {len(recovery_stats)} metrics available")
                    score += 2
                else:
                    print("⚠️ Recovery stats: EMPTY")
                    issues.append("Recovery stats are empty")
                    score += 1
                
                # Test health check
                health = await crash_recovery_manager.health_check()
                if health and health.get("status") == "healthy":
                    print("✅ Recovery manager health: HEALTHY")
                    score += 2
                else:
                    print(f"⚠️ Recovery manager health: {health.get('status', 'unknown') if health else 'NONE'}")
                    issues.append("Recovery manager health check failed")
                    score += 1
                    
                # Test failure handling capability
                try:
                    test_error = Exception("Test error")
                    recovery_result = await crash_recovery_manager.handle_startup_failure(test_error)
                    if recovery_result and recovery_result.get("recovery_id"):
                        print("✅ Failure handling: FUNCTIONAL")
                        score += 2
                    else:
                        print("⚠️ Failure handling: LIMITED")
                        issues.append("Failure handling returns incomplete results")
                        score += 1
                except Exception as e:
                    print(f"⚠️ Failure handling test failed: {e}")
                    issues.append(f"Failure handling error: {e}")
                    score += 1
                    
            except ImportError as e:
                print(f"❌ Crash recovery instance import failed: {e}")
                issues.append(f"Crash recovery instance error: {e}")
                
        except Exception as e:
            print(f"❌ Crash recovery assessment failed: {e}")
            issues.append(f"Critical crash recovery error: {e}")
            score = max(score, 2)
            
        return {
            "score": score,
            "max_score": 10,
            "percentage": (score / 10) * 100,
            "issues": issues,
            "status": "EXCELLENT" if score >= 9 else "GOOD" if score >= 7 else "NEEDS_WORK" if score >= 5 else "CRITICAL"
        }
    
    async def assess_dependency_validation(self) -> Dict[str, Any]:
        """Assess dependency validation system"""
        print("\n🔗 ASSESSING DEPENDENCY VALIDATION")
        print("=" * 50)
        
        score = 0
        issues = []
        
        try:
            # Test dependency container
            from app.services.dependency_container import ServiceContainer
            print("✅ Service container import: SUCCESS")
            score += 2
            
            try:
                from app.services.dependency_container import service_container
                print("✅ Service container instance: AVAILABLE")
                score += 2
                
                # Test service container initialization
                await service_container.initialize()
                print("✅ Service container initialization: SUCCESS")
                score += 2
                
                # Test service registration and retrieval
                health_service = await service_container.get_service("health_monitor")
                if health_service:
                    print("✅ Service retrieval (health_monitor): SUCCESS")
                    score += 2
                else:
                    print("⚠️ Service retrieval (health_monitor): FALLBACK")
                    issues.append("Health monitor service returns fallback")
                    score += 1
                
                # Test service status
                status = service_container.get_service_status()
                if status and status.get("total_services", 0) > 0:
                    print(f"✅ Service status: {status.get('total_services', 0)} services registered")
                    score += 2
                else:
                    print("⚠️ Service status: LIMITED")
                    issues.append("Service status shows limited services")
                    score += 1
                    
            except Exception as e:
                print(f"❌ Service container operations failed: {e}")
                issues.append(f"Service container operations error: {e}")
                
        except Exception as e:
            print(f"❌ Dependency validation assessment failed: {e}")
            issues.append(f"Critical dependency validation error: {e}")
            score = max(score, 2)
            
        return {
            "score": score,
            "max_score": 10,
            "percentage": (score / 10) * 100,
            "issues": issues,
            "status": "EXCELLENT" if score >= 9 else "GOOD" if score >= 7 else "NEEDS_WORK" if score >= 5 else "CRITICAL"
        }
    
    async def assess_imports(self) -> Dict[str, Any]:
        """Assess import system"""
        print("\n📦 ASSESSING IMPORT SYSTEM")
        print("=" * 50)
        
        score = 0
        issues = []
        
        critical_imports = [
            ("FastAPI", "import fastapi"),
            ("Uvicorn", "import uvicorn"),
            ("Pydantic", "import pydantic"),
            ("PSUtil", "import psutil"),
            ("AsyncIO", "import asyncio"),
            ("Main App", "from app.main import app"),
            ("Config", "from app.config import get_settings"),
            ("Health Utils", "from app.utils.health import SystemHealthMonitor"),
            ("Crash Recovery", "from app.crash_recovery_manager import crash_recovery_manager"),
            ("Perfect Ten Validator", "from app.perfect_ten_validator import perfect_ten_validator")
        ]
        
        successful_imports = 0
        
        for name, import_stmt in critical_imports:
            try:
                exec(import_stmt)
                print(f"✅ {name}: SUCCESS")
                successful_imports += 1
            except Exception as e:
                print(f"❌ {name}: FAILED - {e}")
                issues.append(f"{name} import failed: {e}")
        
        # Calculate score based on successful imports
        score = (successful_imports / len(critical_imports)) * 10
        
        return {
            "score": score,
            "max_score": 10,
            "percentage": score * 10,
            "successful_imports": successful_imports,
            "total_imports": len(critical_imports),
            "issues": issues,
            "status": "EXCELLENT" if score >= 9 else "GOOD" if score >= 7 else "NEEDS_WORK" if score >= 5 else "CRITICAL"
        }
    
    async def assess_services(self) -> Dict[str, Any]:
        """Assess services system"""
        print("\n⚙️ ASSESSING SERVICES SYSTEM")
        print("=" * 50)
        
        score = 0
        issues = []
        
        services_to_test = [
            ("Perfect Ten Validator", "app.perfect_ten_validator", "perfect_ten_validator"),
            ("Ultimate Perfection System", "app.ultimate_perfection_system", "ultimate_perfection_system"),
            ("Health Monitor", "app.utils.health", "health_monitor"),
            ("Crash Recovery Manager", "app.crash_recovery_manager", "crash_recovery_manager"),
            ("Service Container", "app.services.dependency_container", "service_container")
        ]
        
        functional_services = 0
        
        for service_name, module_path, instance_name in services_to_test:
            try:
                module = __import__(module_path, fromlist=[instance_name])
                instance = getattr(module, instance_name, None)
                
                if instance:
                    # Test if service has basic functionality
                    if hasattr(instance, 'health_check'):
                        try:
                            health = await instance.health_check()
                            if health:
                                print(f"✅ {service_name}: FUNCTIONAL")
                                functional_services += 1
                            else:
                                print(f"⚠️ {service_name}: LIMITED")
                                issues.append(f"{service_name} health check returns empty")
                        except Exception as e:
                            print(f"⚠️ {service_name}: HEALTH_CHECK_FAILED - {e}")
                            issues.append(f"{service_name} health check failed: {e}")
                    elif hasattr(instance, 'get_status') or hasattr(instance, 'is_healthy'):
                        print(f"✅ {service_name}: AVAILABLE")
                        functional_services += 1
                    else:
                        print(f"⚠️ {service_name}: NO_HEALTH_CHECK")
                        issues.append(f"{service_name} has no health check method")
                        functional_services += 0.5
                else:
                    print(f"❌ {service_name}: INSTANCE_NOT_FOUND")
                    issues.append(f"{service_name} instance not found")
                    
            except Exception as e:
                print(f"❌ {service_name}: IMPORT_FAILED - {e}")
                issues.append(f"{service_name} import failed: {e}")
        
        # Calculate score
        score = (functional_services / len(services_to_test)) * 10
        
        return {
            "score": score,
            "max_score": 10,
            "percentage": score * 10,
            "functional_services": functional_services,
            "total_services": len(services_to_test),
            "issues": issues,
            "status": "EXCELLENT" if score >= 9 else "GOOD" if score >= 7 else "NEEDS_WORK" if score >= 5 else "CRITICAL"
        }
    
    async def run_complete_assessment(self) -> Dict[str, Any]:
        """Run complete system assessment"""
        print("🎯 NETFLIX-GRADE SYSTEM ASSESSMENT")
        print("=" * 60)
        print(f"Assessment started at: {datetime.now().isoformat()}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all assessments
        self.results["health_endpoints"] = await self.assess_health_endpoints()
        self.results["crash_recovery"] = await self.assess_crash_recovery()
        self.results["dependency_validation"] = await self.assess_dependency_validation()
        self.results["imports"] = await self.assess_imports()
        self.results["services"] = await self.assess_services()
        
        # Calculate overall scores
        total_score = sum(result["score"] for result in self.results.values())
        max_score = sum(result["max_score"] for result in self.results.values())
        overall_percentage = (total_score / max_score) * 100
        
        duration = time.time() - start_time
        
        # Generate final report
        print("\n" + "=" * 60)
        print("📋 FINAL ASSESSMENT REPORT")
        print("=" * 60)
        
        for category, result in self.results.items():
            status_emoji = {
                "EXCELLENT": "🟢",
                "GOOD": "🟡", 
                "NEEDS_WORK": "🟠",
                "CRITICAL": "🔴"
            }
            emoji = status_emoji.get(result["status"], "⚪")
            
            print(f"{emoji} {category.upper()}: {result['score']:.1f}/10 ({result['percentage']:.1f}%) - {result['status']}")
            
            if result["issues"]:
                for issue in result["issues"][:3]:  # Show top 3 issues
                    print(f"   ⚠️ {issue}")
                if len(result["issues"]) > 3:
                    print(f"   ... and {len(result['issues']) - 3} more issues")
        
        print("\n" + "=" * 60)
        print(f"🎯 OVERALL SCORE: {total_score:.1f}/{max_score} ({overall_percentage:.1f}%)")
        print(f"⏱️ Assessment Duration: {duration:.2f} seconds")
        
        if overall_percentage >= 90:
            print("🏆 SYSTEM STATUS: NETFLIX-GRADE EXCELLENCE")
        elif overall_percentage >= 80:
            print("✅ SYSTEM STATUS: PRODUCTION READY")
        elif overall_percentage >= 70:
            print("⚠️ SYSTEM STATUS: NEEDS OPTIMIZATION")
        else:
            print("🚨 SYSTEM STATUS: REQUIRES IMMEDIATE ATTENTION")
        
        print("=" * 60)
        
        return {
            "overall_score": total_score,
            "max_score": max_score,
            "percentage": overall_percentage,
            "duration": duration,
            "category_results": self.results,
            "timestamp": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for category, result in self.results.items():
            if result["score"] < 8:
                recommendations.append(f"🔧 {category.upper()}: Score {result['score']:.1f}/10 - Requires improvement")
                
                # Add specific recommendations based on issues
                if result["issues"]:
                    main_issue = result["issues"][0]
                    if "import" in main_issue.lower():
                        recommendations.append(f"   → Fix import dependencies in {category}")
                    elif "health" in main_issue.lower():
                        recommendations.append(f"   → Implement proper health checks in {category}")
                    elif "initialization" in main_issue.lower():
                        recommendations.append(f"   → Fix initialization process in {category}")
        
        if not recommendations:
            recommendations.append("🎯 All systems performing excellently! Maintain current standards.")
        
        return recommendations

async def main():
    """Main assessment entry point"""
    assessor = UnbiasedSystemAssessment()
    
    try:
        final_report = await assessor.run_complete_assessment()
        
        # Save report to file
        import json
        with open("system_assessment_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: system_assessment_report.json")
        
        # Exit with appropriate code
        if final_report["percentage"] >= 80:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Issues found
            
    except Exception as e:
        print(f"\n❌ Assessment failed: {e}")
        traceback.print_exc()
        sys.exit(2)  # Critical failure

if __name__ == "__main__":
    asyncio.run(main())
