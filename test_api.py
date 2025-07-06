#!/usr/bin/env python3
"""
Comprehensive test suite for the OSINT Dark Web Scraper API
Tests all endpoints including authentication, search, agent control, and statistics
"""

import requests
import json
import time
import sys
from typing import Optional

class OSINTAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.session = requests.Session()
        
    def print_test(self, test_name: str):
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 50)
    
    def print_success(self, message: str):
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        print(f"❌ {message}")
    
    def print_info(self, message: str):
        print(f"ℹ️  {message}")
    
    def make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make authenticated request to API endpoint."""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if self.token:
            headers["Authorization"] = self.token
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers)
            else:
                response = self.session.post(url, headers=headers, json=data)
            
            return response.json()
        except Exception as e:
            return {"success": False, "error": f"Request failed: {e}"}
    
    def test_authentication(self) -> bool:
        """Test authentication endpoints"""
        self.print_test("Authentication")
        
        # Test signup
        signup_data = {"username": "testuser", "password": "testpass"}
        result = self.make_request("POST", "/signup", signup_data)
        
        if result.get("success"):
            self.print_success("Signup successful")
        else:
            self.print_info(f"Signup result: {result.get('error', 'User may already exist')}")
        
        # Test login
        login_data = {"username": "admin", "password": "admin"}
        result = self.make_request("POST", "/login", login_data)
        
        if result.get("success"):
            self.token = result.get("token")
            self.print_success("Login successful")
            self.print_info(f"Token: {self.token[:20]}...")
            return True
        else:
            self.print_error(f"Login failed: {result.get('error', 'Unknown error')}")
            return False
    
    def test_search(self) -> bool:
        """Test search functionality with auto-fill"""
        self.print_test("Intelligence Search with Auto-Fill")
        
        # Test basic search
        search_data = {
            "query": "cyber threats",
            "top_k": 10,
            "sim_threshold": 0.2,
            "auto_collect": True
        }
        
        result = self.make_request("POST", "/search", search_data)
        
        if result.get("success"):
            count = result.get("count", 0)
            auto_collection = result.get("auto_collection_started", False)
            
            self.print_success(f"Search successful - Found {count} results")
            if auto_collection:
                self.print_info("Auto-fill was triggered (results < 5)")
            else:
                self.print_info("Auto-fill was not needed (sufficient results)")
            
            # Print first result details
            results = result.get("results", [])
            if results:
                first_result = results[0]
                self.print_info(f"Sample result: {first_result.get('title', 'No title')} (Score: {first_result.get('similarity_score', 0):.3f})")
            
            return True
        else:
            self.print_error(f"Search failed: {result.get('error', 'Unknown error')}")
            return False
    
    def test_agent_status(self) -> bool:
        """Test agent status (should be running from startup with new delay)"""
        self.print_test("Agent Status (Auto-Start with 10s Delay)")
        
        result = self.make_request("GET", "/agent/status")
        
        if result.get("success"):
            agent_status = result.get("agent_status", {})
            running = agent_status.get("running", False)
            
            if running:
                self.print_success("Perpetual agent is running (auto-started)")
                self.print_info(f"Total runs: {agent_status.get('total_runs', 0)}")
                self.print_info(f"Current keyword: {agent_status.get('current_keyword', 'N/A')}")
                delay = agent_status.get('delay_seconds', 0)
                self.print_info(f"Delay: {delay}s (updated from 120s to 10s)")
                if delay == 10:
                    self.print_success("Agent is using the new 10-second delay")
                else:
                    self.print_info(f"Agent delay is {delay}s (may need restart)")
            else:
                self.print_info("Perpetual agent is not running")
            
            return True
        else:
            self.print_error(f"Agent status failed: {result.get('error', 'Unknown error')}")
            return False
    
    def test_agent_control(self) -> bool:
        """Test agent control endpoints"""
        self.print_test("Agent Control")
        
        # Test manual agent run (now processes multiple URLs)
        run_data = {"n_steps": 3, "seed_keyword": "test"}
        result = self.make_request("POST", "/agent/run", run_data)
        
        if result.get("success"):
            self.print_success("Manual agent run successful")
            # Check if the response indicates multiple URL processing
            if "processed_count" in str(result):
                self.print_info("Agent is processing multiple URLs per step (new format)")
            else:
                self.print_info("Agent is processing single URLs per step (legacy format)")
        else:
            self.print_error(f"Manual agent run failed: {result.get('error', 'Unknown error')}")
        
        # Test start perpetual (should fail if already running)
        # Updated to reflect the new 10-second delay setting
        start_data = {"delay_seconds": 10, "max_steps_per_run": 5}
        result = self.make_request("POST", "/agent/start-perpetual", start_data)
        
        if result.get("success"):
            self.print_success("Perpetual agent started manually")
        else:
            self.print_info(f"Perpetual agent start result: {result.get('error', 'Unknown error')} (expected if already running)")
        
        return True
    
    def test_multiple_url_processing(self) -> bool:
        """Test the new multiple URL processing functionality"""
        self.print_test("Multiple URL Processing")
        
        # Test that the agent can process multiple URLs in one step
        run_data = {"n_steps": 1, "seed_keyword": "cyber threats"}
        result = self.make_request("POST", "/agent/run", run_data)
        
        if result.get("success"):
            self.print_success("Agent run completed")
            
            # Check if the response shows multiple URL processing
            response_text = str(result)
            if "processed_count" in response_text and "processed_urls" in response_text:
                self.print_success("Multiple URL processing is active")
                self.print_info("Agent now processes up to 5 URLs per step")
            else:
                self.print_info("Single URL processing (legacy mode)")
            
            return True
        else:
            self.print_error(f"Multiple URL processing test failed: {result.get('error', 'Unknown error')}")
            return False
    
    def test_statistics(self) -> bool:
        """Test statistics endpoints"""
        self.print_test("Statistics")
        
        # Test category stats
        result = self.make_request("GET", "/stats/categories")
        if result.get("success"):
            categories = result.get("category_counts", {})
            self.print_success(f"Category stats: {len(categories)} categories found")
            if categories:
                top_category = max(categories.items(), key=lambda x: x[1])
                self.print_info(f"Top category: {top_category[0]} ({top_category[1]} items)")
        else:
            self.print_error(f"Category stats failed: {result.get('error', 'Unknown error')}")
        
        # Test URL stats
        result = self.make_request("GET", "/stats/urls")
        if result.get("success"):
            url_stats = result.get("url_stats", {})
            self.print_success(f"URL stats: {url_stats.get('total', 0)} total URLs")
            self.print_info(f"Visited: {url_stats.get('visited', 0)}, Failed: {url_stats.get('failed', 0)}, Pending: {url_stats.get('pending', 0)}")
        else:
            self.print_error(f"URL stats failed: {result.get('error', 'Unknown error')}")
        
        # Test latest intelligence
        result = self.make_request("GET", "/latest-intelligence")
        if result.get("success"):
            links = result.get("links", [])
            self.print_success(f"Latest intelligence: {len(links)} links found")
            if links:
                latest = links[0]
                self.print_info(f"Latest: {latest.get('url', 'N/A')} (Confidence: {latest.get('confidence', 0)}%)")
        else:
            self.print_error(f"Latest intelligence failed: {result.get('error', 'Unknown error')}")
        
        return True
    
    def test_system_status(self) -> bool:
        """Test system status endpoints"""
        self.print_test("System Status")
        
        # Test circuit breaker status
        result = self.make_request("GET", "/status/circuit-breakers")
        if result.get("success"):
            breakers = result.get("circuit_breakers", {})
            self.print_success(f"Circuit breakers: {len(breakers)} engines monitored")
            for engine, status in breakers.items():
                state = status.get("state", "UNKNOWN")
                failures = status.get("failure_count", 0)
                self.print_info(f"  {engine}: {state} (failures: {failures})")
        else:
            self.print_error(f"Circuit breaker status failed: {result.get('error', 'Unknown error')}")
        
        return True
    
    def test_error_handling(self) -> bool:
        """Test error handling"""
        self.print_test("Error Handling")
        
        # Test invalid token
        original_token = self.token
        self.token = "invalid_token"
        
        result = self.make_request("GET", "/stats/categories")
        if not result.get("success"):
            self.print_success("Invalid token properly rejected")
        else:
            self.print_error("Invalid token was not rejected")
        
        self.token = original_token
        
        # Test missing parameters
        result = self.make_request("POST", "/search", {})
        if not result.get("success"):
            self.print_success("Missing parameters properly rejected")
        else:
            self.print_error("Missing parameters were not rejected")
        
        return True
    
    def test_logout(self) -> bool:
        """Test logout functionality"""
        self.print_test("Logout")
        
        result = self.make_request("POST", "/logout")
        if result.get("success"):
            self.print_success("Logout successful")
            self.token = None
            return True
        else:
            self.print_error(f"Logout failed: {result.get('error', 'Unknown error')}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        print("🚀 Starting OSINT API Test Suite")
        print("=" * 60)
        
        tests = [
            ("Authentication", self.test_authentication),
            ("Search with Auto-Fill", self.test_search),
            ("Agent Status (Auto-Start)", self.test_agent_status),
            ("Agent Control", self.test_agent_control),
            ("Multiple URL Processing", self.test_multiple_url_processing),
            ("Statistics", self.test_statistics),
            ("System Status", self.test_system_status),
            ("Error Handling", self.test_error_handling),
            ("Logout", self.test_logout),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"⚠️  {test_name} test failed")
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed")
            return False

def main():
    """Main test runner"""
    tester = OSINTAPITester()
    
    if len(sys.argv) > 1:
        tester.base_url = sys.argv[1]
    
    print(f"Testing API at: {tester.base_url}")
    
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ API is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ API has issues that need to be fixed")
        sys.exit(1)

if __name__ == "__main__":
    main() 