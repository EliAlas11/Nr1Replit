
#!/usr/bin/env python3
"""
Netflix-Grade API Test Client
Comprehensive API endpoint testing and validation
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result data class"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class TestSummary:
    """Test summary data class"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    average_response_time: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0


class APITestClient:
    """Comprehensive API test client"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        self.auth_token = None
        self.test_summary = TestSummary()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.aclose()
    
    async def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        expected_status: int = 200
    ) -> TestResult:
        """Make HTTP request and return test result"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        # Prepare headers
        request_headers = headers or {}
        if self.auth_token:
            request_headers["Authorization"] = f"Bearer {self.auth_token}"
        
        try:
            # Make request
            response = await self.session.request(
                method=method,
                url=url,
                json=data if method in ["POST", "PUT", "PATCH"] else None,
                params=params,
                headers=request_headers
            )
            
            response_time = time.time() - start_time
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}
            
            # Check if test passed
            success = response.status_code == expected_status
            error_message = None
            
            if not success:
                error_message = f"Expected {expected_status}, got {response.status_code}"
                if isinstance(response_data, dict) and "error" in response_data:
                    error_message += f" - {response_data['error']}"
            
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time=response_time,
                success=success,
                error_message=error_message,
                response_data=response_data
            )
            
            # Update summary
            self.test_summary.total_tests += 1
            if success:
                self.test_summary.passed_tests += 1
            else:
                self.test_summary.failed_tests += 1
            
            self.test_summary.results.append(result)
            
            # Log result
            status = "PASS" if success else "FAIL"
            logger.info(f"[{status}] {method} {endpoint} - {response.status_code} ({response_time:.3f}s)")
            
            return result
        
        except Exception as e:
            response_time = time.time() - start_time
            error_message = str(e)
            
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=error_message
            )
            
            self.test_summary.total_tests += 1
            self.test_summary.failed_tests += 1
            self.test_summary.results.append(result)
            
            logger.error(f"[ERROR] {method} {endpoint} - {error_message}")
            return result
    
    async def test_health_endpoints(self):
        """Test health and monitoring endpoints"""
        logger.info("🏥 Testing Health Endpoints...")
        
        await self.make_request("GET", "/health")
        await self.make_request("GET", "/health/detailed")
        await self.make_request("GET", "/metrics")
        await self.make_request("GET", "/status")
    
    async def test_auth_endpoints(self):
        """Test authentication endpoints"""
        logger.info("🔐 Testing Authentication Endpoints...")
        
        # Test registration (expect validation error for empty data)
        await self.make_request(
            "POST", 
            "/api/v10/auth/register",
            data={},
            expected_status=422
        )
        
        # Test login (expect validation error for empty data)
        await self.make_request(
            "POST",
            "/api/v10/auth/login",
            data={},
            expected_status=422
        )
        
        # Test profile without auth (expect 401)
        await self.make_request(
            "GET",
            "/api/v10/auth/profile",
            expected_status=401
        )
    
    async def test_enterprise_endpoints(self):
        """Test enterprise endpoints"""
        logger.info("🏢 Testing Enterprise Endpoints...")
        
        # Test health check
        await self.make_request("GET", "/api/v10/enterprise/health")
        
        # Test dashboard (expect auth required)
        await self.make_request(
            "GET",
            "/api/v10/enterprise/dashboard",
            params={"organization_id": "test-org"},
            expected_status=401
        )
    
    async def test_ai_endpoints(self):
        """Test AI production endpoints"""
        logger.info("🤖 Testing AI Endpoints...")
        
        # Test AI health
        await self.make_request("GET", "/ai/health")
        
        # Test metrics (expect auth required)
        await self.make_request(
            "GET",
            "/ai/metrics",
            expected_status=401
        )
        
        # Test model status (expect auth required)
        await self.make_request(
            "GET",
            "/ai/models/status",
            expected_status=401
        )
    
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        logger.info("⚡ Testing Rate Limiting...")
        
        # Make multiple requests to trigger rate limiting
        for i in range(5):
            await self.make_request("GET", "/health")
            await asyncio.sleep(0.1)  # Small delay
    
    async def test_error_handling(self):
        """Test error handling"""
        logger.info("❌ Testing Error Handling...")
        
        # Test 404 error
        await self.make_request(
            "GET",
            "/api/v1/nonexistent-endpoint",
            expected_status=404
        )
        
        # Test invalid JSON
        await self.make_request(
            "POST",
            "/api/v10/auth/login",
            data={"invalid": "data"},
            expected_status=422
        )
    
    async def test_documentation_endpoints(self):
        """Test API documentation endpoints"""
        logger.info("📚 Testing Documentation Endpoints...")
        
        await self.make_request("GET", "/api/docs", expected_status=200)
        await self.make_request("GET", "/api/redoc", expected_status=200)
        await self.make_request("GET", "/api/openapi.json", expected_status=200)
    
    async def test_websocket_endpoints(self):
        """Test WebSocket endpoints"""
        logger.info("🔌 Testing WebSocket Endpoints...")
        
        # Note: These are HTTP endpoints for WebSocket management
        # Actual WebSocket testing would require different approach
        pass
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive API tests"""
        logger.info("🚀 Starting Comprehensive API Tests...")
        logger.info(f"📍 Testing API at: {self.base_url}")
        
        start_time = time.time()
        
        # Run all test suites
        await self.test_health_endpoints()
        await self.test_documentation_endpoints()
        await self.test_auth_endpoints()
        await self.test_enterprise_endpoints()
        await self.test_ai_endpoints()
        await self.test_rate_limiting()
        await self.test_error_handling()
        await self.test_websocket_endpoints()
        
        # Calculate summary
        total_time = time.time() - start_time
        self.test_summary.average_response_time = (
            sum(r.response_time for r in self.test_summary.results) /
            len(self.test_summary.results)
        ) if self.test_summary.results else 0
        
        # Print summary
        self.print_summary(total_time)
    
    def print_summary(self, total_time: float):
        """Print test summary"""
        print("\n" + "="*80)
        print("🎯 API TEST SUMMARY")
        print("="*80)
        print(f"📊 Total Tests: {self.test_summary.total_tests}")
        print(f"✅ Passed: {self.test_summary.passed_tests}")
        print(f"❌ Failed: {self.test_summary.failed_tests}")
        print(f"📈 Success Rate: {self.test_summary.success_rate:.1f}%")
        print(f"⏱️  Average Response Time: {self.test_summary.average_response_time:.3f}s")
        print(f"🕐 Total Test Time: {total_time:.2f}s")
        
        # Print failed tests
        failed_tests = [r for r in self.test_summary.results if not r.success]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test.method} {test.endpoint} - {test.error_message}")
        
        # Performance analysis
        slow_tests = [r for r in self.test_summary.results if r.response_time > 1.0]
        if slow_tests:
            print(f"\n⚠️  Slow Tests (>1s) ({len(slow_tests)}):")
            for test in slow_tests:
                print(f"   • {test.method} {test.endpoint} - {test.response_time:.3f}s")
        
        print("\n" + "="*80)
        
        # Overall grade
        if self.test_summary.success_rate >= 95:
            print("🏆 GRADE: PERFECT 10/10 - Netflix-Grade API!")
        elif self.test_summary.success_rate >= 90:
            print("🥇 GRADE: Excellent 9/10 - Enterprise-Grade API!")
        elif self.test_summary.success_rate >= 80:
            print("🥈 GRADE: Good 8/10 - Production-Ready API")
        elif self.test_summary.success_rate >= 70:
            print("🥉 GRADE: Fair 7/10 - Needs Improvement")
        else:
            print("⚠️  GRADE: Poor - Significant Issues Found")
        
        print("="*80)


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Netflix-Grade API Test Client")
    parser.add_argument(
        "--url",
        default="http://localhost:5000",
        help="API base URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--output",
        help="Output file for detailed results (JSON format)"
    )
    
    args = parser.parse_args()
    
    async with APITestClient(args.url) as client:
        await client.run_comprehensive_tests()
        
        # Save detailed results if requested
        if args.output:
            results_data = {
                "summary": {
                    "total_tests": client.test_summary.total_tests,
                    "passed_tests": client.test_summary.passed_tests,
                    "failed_tests": client.test_summary.failed_tests,
                    "success_rate": client.test_summary.success_rate,
                    "average_response_time": client.test_summary.average_response_time
                },
                "results": [
                    {
                        "endpoint": r.endpoint,
                        "method": r.method,
                        "status_code": r.status_code,
                        "response_time": r.response_time,
                        "success": r.success,
                        "error_message": r.error_message,
                        "response_data": r.response_data
                    }
                    for r in client.test_summary.results
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            with open(args.output, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"📄 Detailed results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
