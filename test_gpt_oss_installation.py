#!/usr/bin/env python3
"""
GPT-OSS-20B Installation Test Suite
Comprehensive end-to-end testing for Apple Silicon installation
"""

import sys
import time
import json
import subprocess
import psutil
import torch
from datetime import datetime

class GPTOSSTestSuite:
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.test_prompts = [
            {
                "name": "basic_reasoning",
                "prompt": "Explain why the sky appears blue in simple terms.",
                "expected_tokens": 50
            },
            {
                "name": "code_generation", 
                "prompt": "Write a Python function to calculate fibonacci numbers recursively.",
                "expected_tokens": 80
            },
            {
                "name": "complex_reasoning",
                "prompt": "Compare and contrast renewable vs non-renewable energy sources, including pros and cons.",
                "expected_tokens": 150
            }
        ]
    
    def log(self, message, test_name=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}]"
        if test_name:
            prefix += f" [{test_name}]"
        print(f"{prefix} {message}")
    
    def test_system_requirements(self):
        """Test system requirements and hardware compatibility"""
        self.log("Testing system requirements...", "SYSTEM")
        
        results = {}
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        results['ram_gb'] = round(ram_gb, 1)
        results['ram_sufficient'] = ram_gb >= 16
        self.log(f"RAM: {results['ram_gb']}GB ({'✓' if results['ram_sufficient'] else '✗'})")
        
        # Check disk space
        disk_free = psutil.disk_usage('/').free / (1024**3)
        results['disk_free_gb'] = round(disk_free, 1)
        results['disk_sufficient'] = disk_free >= 20
        self.log(f"Free disk space: {results['disk_free_gb']}GB ({'✓' if results['disk_sufficient'] else '✗'})")
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        results['python_version'] = python_version
        results['python_compatible'] = sys.version_info >= (3, 10)
        self.log(f"Python version: {python_version} ({'✓' if results['python_compatible'] else '✗'})")
        
        return results
    
    def test_pytorch_mps(self):
        """Test PyTorch MPS (Apple Silicon GPU) support"""
        self.log("Testing PyTorch MPS support...", "PYTORCH")
        
        results = {}
        
        try:
            results['torch_version'] = torch.__version__
            results['mps_available'] = torch.backends.mps.is_available()
            results['mps_built'] = torch.backends.mps.is_built()
            
            self.log(f"PyTorch version: {results['torch_version']}")
            self.log(f"MPS available: {'✓' if results['mps_available'] else '✗'}")
            self.log(f"MPS built: {'✓' if results['mps_built'] else '✗'}")
            
            if results['mps_available']:
                # Test MPS functionality
                device = torch.device("mps")
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.mm(test_tensor, test_tensor.t())
                results['mps_functional'] = True
                self.log("MPS GPU computation test: ✓")
            else:
                results['mps_functional'] = False
                
        except Exception as e:
            results['error'] = str(e)
            results['mps_functional'] = False
            self.log(f"PyTorch MPS test failed: {e}")
            
        return results
    
    def test_lm_studio_process(self):
        """Test LM Studio process and resource usage"""
        self.log("Testing LM Studio processes...", "LM_STUDIO")
        
        results = {}
        lm_studio_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'LM Studio' in proc.info['name']:
                    lm_studio_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_mb': round(proc.info['memory_info'].rss / (1024**2), 1),
                        'cpu_percent': proc.info['cpu_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        results['processes'] = lm_studio_processes
        results['is_running'] = len(lm_studio_processes) > 0
        
        if results['is_running']:
            total_memory = sum(p['memory_mb'] for p in lm_studio_processes)
            results['total_memory_mb'] = round(total_memory, 1)
            results['total_memory_gb'] = round(total_memory / 1024, 1)
            self.log(f"LM Studio running: ✓ ({len(lm_studio_processes)} processes)")
            self.log(f"Total memory usage: {results['total_memory_gb']}GB")
        else:
            self.log("LM Studio not running: ✗")
            
        return results
    
    def test_huggingface_cli(self):
        """Test Hugging Face CLI functionality"""
        self.log("Testing Hugging Face CLI...", "HF_CLI")
        
        results = {}
        
        try:
            # Test CLI availability
            result = subprocess.run(['huggingface-cli', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            results['cli_available'] = result.returncode == 0
            
            if results['cli_available']:
                self.log("Hugging Face CLI: ✓")
                
                # Test model info retrieval
                model_result = subprocess.run([
                    'huggingface-cli', 'scan-cache', '--verbose'
                ], capture_output=True, text=True, timeout=30)
                
                results['cache_scan_success'] = model_result.returncode == 0
                if 'gpt-oss' in model_result.stdout.lower():
                    results['model_in_cache'] = True
                    self.log("GPT-OSS model found in cache: ✓")
                else:
                    results['model_in_cache'] = False
                    self.log("GPT-OSS model not in cache: ✗")
                    
            else:
                self.log("Hugging Face CLI: ✗")
                
        except Exception as e:
            results['error'] = str(e)
            results['cli_available'] = False
            self.log(f"HF CLI test failed: {e}")
            
        return results
    
    def test_transformers_loading(self):
        """Test model loading with Transformers library"""
        self.log("Testing Transformers model loading...", "TRANSFORMERS")
        
        results = {}
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.log("Loading tokenizer...")
            start_time = time.time()
            
            # Test tokenizer loading
            tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            tokenizer_time = time.time() - start_time
            results['tokenizer_load_time'] = round(tokenizer_time, 2)
            results['tokenizer_loaded'] = True
            self.log(f"Tokenizer loaded: ✓ ({tokenizer_time:.2f}s)")
            
            # Test a simple tokenization
            test_text = "Hello, how are you today?"
            tokens = tokenizer.encode(test_text)
            results['tokenization_works'] = len(tokens) > 0
            results['sample_token_count'] = len(tokens)
            self.log(f"Tokenization test: ✓ ({len(tokens)} tokens)")
            
            # Note: We won't load the full model in testing to avoid long waits
            # but we verify the model files are accessible
            try:
                model_info = AutoModelForCausalLM.from_pretrained(
                    "openai/gpt-oss-20b", 
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                results['model_accessible'] = True
                results['model_loaded'] = True
                self.log("Model loading: ✓ (Basic verification)")
                
                # Clean up immediately to save memory
                del model_info
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as model_e:
                results['model_accessible'] = False
                results['model_loaded'] = False
                results['model_error'] = str(model_e)
                self.log(f"Model loading failed: {model_e}")
                
        except Exception as e:
            results['error'] = str(e)
            results['tokenizer_loaded'] = False
            results['model_loaded'] = False
            self.log(f"Transformers test failed: {e}")
            
        return results
    
    def test_model_inference_simulation(self):
        """Simulate model inference to test setup"""
        self.log("Testing model inference simulation...", "INFERENCE")
        
        results = {}
        
        try:
            # Create a simple torch model simulation
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                self.log("Using MPS device for inference simulation")
            else:
                device = torch.device("cpu")
                self.log("Using CPU device for inference simulation")
                
            # Simulate model loading time and memory usage
            start_time = time.time()
            
            # Create a tensor to simulate model parameters (~1GB)
            param_tensor = torch.randn(131072, 1000, device=device, dtype=torch.float16)
            
            # Simulate inference
            input_tensor = torch.randn(1, 1000, device=device, dtype=torch.float16)
            with torch.no_grad():
                output = torch.mm(input_tensor, param_tensor.t())
                
            inference_time = time.time() - start_time
            results['simulation_time'] = round(inference_time, 3)
            results['device_used'] = str(device)
            results['simulation_success'] = True
            
            self.log(f"Inference simulation: ✓ ({inference_time:.3f}s on {device})")
            
            # Clean up
            del param_tensor, input_tensor, output
            if device.type == "mps":
                torch.mps.empty_cache()
                
        except Exception as e:
            results['error'] = str(e)
            results['simulation_success'] = False
            self.log(f"Inference simulation failed: {e}")
            
        return results
    
    def performance_benchmark(self):
        """Run performance benchmarks"""
        self.log("Running performance benchmarks...", "BENCHMARK")
        
        results = {}
        
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                
            # Matrix multiplication benchmark (simulates transformer operations)
            sizes = [512, 1024, 2048]
            for size in sizes:
                self.log(f"Testing {size}x{size} matrix multiplication...")
                
                start_time = time.time()
                a = torch.randn(size, size, device=device, dtype=torch.float16)
                b = torch.randn(size, size, device=device, dtype=torch.float16)
                
                # Warm up
                for _ in range(3):
                    torch.mm(a, b)
                
                # Benchmark
                start_bench = time.time()
                for _ in range(10):
                    c = torch.mm(a, b)
                end_bench = time.time()
                
                avg_time = (end_bench - start_bench) / 10
                results[f'matmul_{size}x{size}_avg_ms'] = round(avg_time * 1000, 2)
                
                self.log(f"{size}x{size} matmul: {avg_time*1000:.2f}ms average")
                
                del a, b, c
                
            if device.type == "mps":
                torch.mps.empty_cache()
                
            results['benchmark_success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['benchmark_success'] = False
            self.log(f"Performance benchmark failed: {e}")
            
        return results
    
    def run_all_tests(self):
        """Run complete test suite"""
        self.start_time = time.time()
        self.log("Starting GPT-OSS-20B Installation Test Suite", "START")
        self.log("=" * 60)
        
        # Run all tests
        test_results = {}
        
        test_results['system'] = self.test_system_requirements()
        test_results['pytorch'] = self.test_pytorch_mps()
        test_results['lm_studio'] = self.test_lm_studio_process()
        test_results['huggingface'] = self.test_huggingface_cli()
        test_results['transformers'] = self.test_transformers_loading()
        test_results['inference'] = self.test_model_inference_simulation()
        test_results['performance'] = self.performance_benchmark()
        
        # Calculate overall status
        total_time = time.time() - self.start_time
        test_results['meta'] = {
            'total_time_seconds': round(total_time, 2),
            'timestamp': datetime.now().isoformat(),
            'platform': 'Apple Silicon macOS'
        }
        
        self.results = test_results
        
        # Generate summary
        self.generate_test_summary()
        
        return test_results
    
    def generate_test_summary(self):
        """Generate test summary report"""
        self.log("=" * 60)
        self.log("TEST SUMMARY REPORT", "SUMMARY")
        self.log("=" * 60)
        
        # System Requirements
        sys_results = self.results['system']
        self.log(f"System Requirements: {'✓ PASS' if all([sys_results['ram_sufficient'], sys_results['disk_sufficient'], sys_results['python_compatible']]) else '✗ FAIL'}")
        
        # PyTorch MPS
        torch_results = self.results['pytorch']
        self.log(f"PyTorch MPS Support: {'✓ PASS' if torch_results.get('mps_functional', False) else '✗ FAIL'}")
        
        # LM Studio
        lm_results = self.results['lm_studio']
        self.log(f"LM Studio Process: {'✓ RUNNING' if lm_results['is_running'] else '✗ NOT RUNNING'}")
        
        # Hugging Face
        hf_results = self.results['huggingface']
        self.log(f"Hugging Face CLI: {'✓ AVAILABLE' if hf_results.get('cli_available', False) else '✗ UNAVAILABLE'}")
        
        # Transformers
        trans_results = self.results['transformers']
        self.log(f"Transformers Library: {'✓ WORKING' if trans_results.get('tokenizer_loaded', False) else '✗ ISSUES'}")
        
        # Inference
        inf_results = self.results['inference']
        self.log(f"Inference Simulation: {'✓ PASS' if inf_results.get('simulation_success', False) else '✗ FAIL'}")
        
        # Performance
        perf_results = self.results['performance']
        self.log(f"Performance Benchmark: {'✓ PASS' if perf_results.get('benchmark_success', False) else '✗ FAIL'}")
        
        self.log("=" * 60)
        self.log(f"Total test time: {self.results['meta']['total_time_seconds']}s")
        
        # Recommendations
        self.log("\nRECOMMENDATIONS:", "SUMMARY")
        
        if not torch_results.get('mps_functional', False):
            self.log("⚠️  MPS not working - model will run on CPU (slower)")
            
        if not lm_results['is_running']:
            self.log("⚠️  LM Studio not running - start it to use GUI interface")
            
        if lm_results.get('total_memory_gb', 0) > 20:
            self.log("⚠️  High memory usage detected - monitor system resources")
            
        if not hf_results.get('model_in_cache', False):
            self.log("ℹ️  GPT-OSS model not found in HF cache - may need to download")

if __name__ == "__main__":
    test_suite = GPTOSSTestSuite()
    results = test_suite.run_all_tests()
    
    # Save results to file
    with open('gpt_oss_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: gpt_oss_test_results.json")