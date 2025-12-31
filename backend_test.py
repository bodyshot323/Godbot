#!/usr/bin/env python3

import requests
import sys
import json
from datetime import datetime

class GodBotAPITester:
    def __init__(self, base_url="https://godbot-prototype.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.session_id = None
        self.persona_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if not endpoint.startswith('http') else endpoint
        default_headers = {'Content-Type': 'application/json'}
        if headers:
            default_headers.update(headers)

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=default_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=default_headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=default_headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API", "GET", "", 200)

    def test_system_status(self):
        """Test system status endpoint"""
        success, data = self.run_test("System Status", "GET", "status", 200)
        if success:
            required_fields = ['status', 'fusion_mode', 'active_models', 'db_connected']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing required field: {field}")
                    return False
            print(f"   Fusion Mode: {data.get('fusion_mode')}")
            print(f"   DB Connected: {data.get('db_connected')}")
            print(f"   Active Models: {len(data.get('active_models', []))}")
        return success

    def test_pledge_endpoint(self):
        """Test GodBot pledge endpoint"""
        success, data = self.run_test("GodBot Pledge", "GET", "pledge", 200)
        if success:
            required_fields = ['pledge', 'version', 'codename', 'principles']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing required field: {field}")
                    return False
            print(f"   Codename: {data.get('codename')}")
            print(f"   Version: {data.get('version')}")
        return success

    def test_dashboard_endpoint(self):
        """Test dashboard metrics endpoint"""
        success, data = self.run_test("Dashboard Metrics", "GET", "dashboard", 200)
        if success:
            required_fields = ['usage', 'tier_info', 'model_breakdown', 'cost_comparison']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing required field: {field}")
                    return False
            print(f"   Tier: {data.get('tier_info', {}).get('name')}")
            print(f"   Credits Remaining: {data.get('usage', {}).get('credits_remaining')}")
            print(f"   Models: {len(data.get('model_breakdown', []))}")
        return success

    def test_dreamchain_endpoint(self):
        """Test DreamChain insights endpoint"""
        success, data = self.run_test("DreamChain Insights", "GET", "dreamchain", 200)
        if success:
            required_fields = ['mode', 'status', 'insights']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing required field: {field}")
                    return False
            print(f"   Mode: {data.get('mode')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Insights: {len(data.get('insights', []))}")
        return success

    def test_personas_endpoint(self):
        """Test personas endpoint"""
        success, data = self.run_test("Get Personas", "GET", "personas", 200)
        if success:
            if not isinstance(data, list):
                print("❌ Personas response should be a list")
                return False
            if len(data) < 4:
                print(f"❌ Expected at least 4 default personas, got {len(data)}")
                return False
            
            # Check for default personas
            persona_names = [p.get('name') for p in data]
            expected_personas = ['GODMIND', 'LUMINA', 'SENTINEL', 'MAGGIE']
            for expected in expected_personas:
                if expected not in persona_names:
                    print(f"❌ Missing default persona: {expected}")
                    return False
            
            # Store first persona for chat test
            self.persona_id = data[0].get('id')
            print(f"   Found {len(data)} personas: {persona_names}")
        return success

    def test_tiers_endpoint(self):
        """Test tiers configuration endpoint"""
        success, data = self.run_test("Get Tiers", "GET", "tiers", 200)
        if success:
            expected_tiers = ['free', 'pro', 'dev', 'god']
            for tier in expected_tiers:
                if tier not in data:
                    print(f"❌ Missing tier configuration: {tier}")
                    return False
            print(f"   Available tiers: {list(data.keys())}")
        return success

    def test_chat_endpoint(self):
        """Test chat endpoint with fallback response"""
        chat_data = {
            "message": "Hello, test message for GodBot",
            "persona_id": self.persona_id or "godmind-default",
            "tier": "dev"
        }
        
        success, data = self.run_test("Chat Message", "POST", "chat", 200, chat_data)
        if success:
            required_fields = ['id', 'session_id', 'content', 'fusion_mode', 'models_used']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing required field: {field}")
                    return False
            
            self.session_id = data.get('session_id')
            print(f"   Session ID: {self.session_id}")
            print(f"   Fusion Mode: {data.get('fusion_mode')}")
            print(f"   Models Used: {data.get('models_used')}")
            print(f"   Response Length: {len(data.get('content', ''))}")
        return success

    def test_sessions_endpoint(self):
        """Test sessions endpoint"""
        success, data = self.run_test("Get Sessions", "GET", "sessions", 200)
        if success:
            if not isinstance(data, list):
                print("❌ Sessions response should be a list")
                return False
            print(f"   Found {len(data)} sessions")
            
            # If we have a session from chat test, verify it exists
            if self.session_id:
                session_ids = [s.get('id') for s in data]
                if self.session_id not in session_ids:
                    print(f"❌ Session {self.session_id} not found in sessions list")
                    return False
        return success

    def test_session_messages(self):
        """Test getting messages for a session"""
        if not self.session_id:
            print("⚠️  Skipping session messages test - no session ID available")
            return True
            
        success, data = self.run_test(
            "Get Session Messages", 
            "GET", 
            f"sessions/{self.session_id}/messages", 
            200
        )
        if success:
            if not isinstance(data, list):
                print("❌ Messages response should be a list")
                return False
            print(f"   Found {len(data)} messages in session")
        return success

    def test_delete_session(self):
        """Test deleting a session"""
        if not self.session_id:
            print("⚠️  Skipping session deletion test - no session ID available")
            return True
            
        success, data = self.run_test(
            "Delete Session", 
            "DELETE", 
            f"sessions/{self.session_id}", 
            200
        )
        return success

def main():
    print("🚀 Starting GodBot API Tests...")
    print("=" * 60)
    
    tester = GodBotAPITester()
    
    # Test sequence
    tests = [
        ("Root Endpoint", tester.test_root_endpoint),
        ("System Status", tester.test_system_status),
        ("GodBot Pledge", tester.test_pledge_endpoint),
        ("Dashboard Metrics", tester.test_dashboard_endpoint),
        ("DreamChain Insights", tester.test_dreamchain_endpoint),
        ("Personas", tester.test_personas_endpoint),
        ("Tiers Configuration", tester.test_tiers_endpoint),
        ("Chat Message", tester.test_chat_endpoint),
        ("Sessions List", tester.test_sessions_endpoint),
        ("Session Messages", tester.test_session_messages),
        ("Delete Session", tester.test_delete_session),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            failed_tests.append(test_name)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tester.tests_passed}/{tester.tests_run} passed")
    
    if failed_tests:
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())