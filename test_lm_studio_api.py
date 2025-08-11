#!/usr/bin/env python3
"""
LM Studio API Testing
Tests the local LM Studio server API functionality
"""

import requests
import json
import time
from datetime import datetime

class LMStudioAPITest:
    def __init__(self, base_url="http://localhost:1234"):
        self.base_url = base_url
        self.results = {}
    
    def log(self, message, test_name=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}]"
        if test_name:
            prefix += f" [{test_name}]"
        print(f"{prefix} {message}")
    
    def test_server_status(self):
        """Test if LM Studio server is running and accessible"""
        self.log("Testing LM Studio server status...", "API")
        
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            
            if response.status_code == 200:
                models = response.json()
                self.results['server_accessible'] = True
                self.results['models_available'] = len(models.get('data', []))
                self.log(f"Server accessible: ✓ ({self.results['models_available']} models)")
                
                # Check if GPT-OSS model is loaded
                gpt_oss_loaded = any('gpt-oss' in model.get('id', '').lower() 
                                   for model in models.get('data', []))
                self.results['gpt_oss_loaded'] = gpt_oss_loaded
                
                if gpt_oss_loaded:
                    self.log("GPT-OSS model loaded: ✓")
                else:
                    self.log("GPT-OSS model not loaded: ✗")
                
                return models
                
            else:
                self.results['server_accessible'] = False
                self.log(f"Server not accessible: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.results['server_accessible'] = False
            self.results['error'] = str(e)
            self.log(f"Server connection failed: {e}")
            return None
    
    def test_chat_completion(self):
        """Test chat completion API"""
        if not self.results.get('server_accessible', False):
            self.log("Skipping chat test - server not accessible", "CHAT")
            return None
            
        self.log("Testing chat completion...", "CHAT")
        
        try:
            payload = {
                "model": "gpt-oss-20b",  # This might need adjustment based on actual model name
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Please give a brief answer."
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.results['chat_success'] = True
                self.results['response_time'] = round(response_time, 2)
                
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    self.results['response_content'] = content
                    self.results['response_length'] = len(content)
                    
                    self.log(f"Chat completion: ✓ ({response_time:.2f}s)")
                    self.log(f"Response: {content[:100]}...")
                    
                    # Calculate tokens per second estimate
                    if 'usage' in data:
                        total_tokens = data['usage'].get('total_tokens', 0)
                        if total_tokens > 0:
                            tokens_per_second = total_tokens / response_time
                            self.results['tokens_per_second'] = round(tokens_per_second, 2)
                            self.log(f"Performance: ~{tokens_per_second:.2f} tokens/second")
                
                return data
            else:
                self.results['chat_success'] = False
                self.results['chat_error'] = f"HTTP {response.status_code}: {response.text}"
                self.log(f"Chat completion failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.results['chat_success'] = False
            self.results['chat_error'] = str(e)
            self.log(f"Chat completion error: {e}")
            return None
    
    def test_model_info(self):
        """Get detailed model information"""
        if not self.results.get('server_accessible', False):
            return None
            
        self.log("Getting model information...", "MODEL_INFO")
        
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            
            if response.status_code == 200:
                models_data = response.json()
                self.results['model_info'] = models_data
                
                for model in models_data.get('data', []):
                    model_id = model.get('id', 'Unknown')
                    self.log(f"Available model: {model_id}")
                
                return models_data
            else:
                self.log(f"Failed to get model info: {response.status_code}")
                return None
                
        except Exception as e:
            self.log(f"Model info error: {e}")
            return None
    
    def run_all_tests(self):
        """Run all LM Studio API tests"""
        self.log("Starting LM Studio API Tests", "START")
        self.log("=" * 50)
        
        # Test server status and models
        models = self.test_server_status()
        
        # Get detailed model info
        self.test_model_info()
        
        # Test chat completion
        self.test_chat_completion()
        
        # Generate summary
        self.generate_api_summary()
        
        return self.results
    
    def generate_api_summary(self):
        """Generate API test summary"""
        self.log("=" * 50)
        self.log("LM STUDIO API TEST SUMMARY", "SUMMARY")
        self.log("=" * 50)
        
        server_status = "✓ ACCESSIBLE" if self.results.get('server_accessible', False) else "✗ NOT ACCESSIBLE"
        self.log(f"Server Status: {server_status}")
        
        if self.results.get('models_available', 0) > 0:
            self.log(f"Models Available: ✓ ({self.results['models_available']})")
        else:
            self.log("Models Available: ✗ (0)")
        
        gpt_oss_status = "✓ LOADED" if self.results.get('gpt_oss_loaded', False) else "✗ NOT LOADED"
        self.log(f"GPT-OSS Model: {gpt_oss_status}")
        
        chat_status = "✓ WORKING" if self.results.get('chat_success', False) else "✗ FAILED"
        self.log(f"Chat Completion: {chat_status}")
        
        if self.results.get('response_time'):
            self.log(f"Response Time: {self.results['response_time']}s")
        
        if self.results.get('tokens_per_second'):
            self.log(f"Performance: {self.results['tokens_per_second']} tokens/second")
        
        self.log("=" * 50)

if __name__ == "__main__":
    api_test = LMStudioAPITest()
    results = api_test.run_all_tests()
    
    # Save results
    with open('lm_studio_api_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAPI test results saved to: lm_studio_api_test_results.json")