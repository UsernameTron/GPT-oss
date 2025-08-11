#!/usr/bin/env python3
"""
Direct Model Inference Test
Tests actual GPT-OSS-20B model loading and inference
"""

import time
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import psutil
import gc

class ModelInferenceTest:
    def __init__(self):
        self.results = {}
        self.model = None
        self.tokenizer = None
    
    def log(self, message, test_name=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}]"
        if test_name:
            prefix += f" [{test_name}]"
        print(f"{prefix} {message}")
    
    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process()
        return {
            'rss_gb': round(process.memory_info().rss / (1024**3), 2),
            'vms_gb': round(process.memory_info().vms / (1024**3), 2),
            'percent': round(process.memory_percent(), 2)
        }
    
    def test_tokenizer_loading(self):
        """Test tokenizer loading"""
        self.log("Testing tokenizer loading...", "TOKENIZER")
        
        try:
            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            load_time = time.time() - start_time
            
            self.results['tokenizer'] = {
                'loaded': True,
                'load_time': round(load_time, 2),
                'vocab_size': len(self.tokenizer) if self.tokenizer else 0
            }
            
            self.log(f"Tokenizer loaded: ✓ ({load_time:.2f}s, vocab: {self.results['tokenizer']['vocab_size']})")
            
            # Test tokenization
            test_text = "Hello, how are you today? This is a test."
            tokens = self.tokenizer.encode(test_text)
            decoded = self.tokenizer.decode(tokens)
            
            self.results['tokenizer']['test_tokens'] = len(tokens)
            self.results['tokenizer']['decode_match'] = test_text.strip() == decoded.strip()
            
            self.log(f"Tokenization test: ✓ ({len(tokens)} tokens)")
            
            return True
            
        except Exception as e:
            self.results['tokenizer'] = {
                'loaded': False,
                'error': str(e)
            }
            self.log(f"Tokenizer loading failed: {e}")
            return False
    
    def test_model_loading_light(self):
        """Test lightweight model loading (config only)"""
        self.log("Testing model configuration loading...", "MODEL_CONFIG")
        
        try:
            from transformers import AutoConfig
            
            start_time = time.time()
            config = AutoConfig.from_pretrained("openai/gpt-oss-20b")
            load_time = time.time() - start_time
            
            self.results['model_config'] = {
                'loaded': True,
                'load_time': round(load_time, 2),
                'num_parameters': getattr(config, 'num_parameters', 'unknown'),
                'hidden_size': getattr(config, 'hidden_size', 'unknown'),
                'num_attention_heads': getattr(config, 'num_attention_heads', 'unknown'),
                'num_hidden_layers': getattr(config, 'num_hidden_layers', 'unknown'),
                'vocab_size': getattr(config, 'vocab_size', 'unknown')
            }
            
            self.log(f"Model config loaded: ✓ ({load_time:.2f}s)")
            self.log(f"Architecture: {self.results['model_config']['hidden_size']} hidden, {self.results['model_config']['num_hidden_layers']} layers")
            
            return True
            
        except Exception as e:
            self.results['model_config'] = {
                'loaded': False,
                'error': str(e)
            }
            self.log(f"Model config loading failed: {e}")
            return False
    
    def test_model_loading_full(self):
        """Test full model loading (WARNING: This will use significant memory)"""
        self.log("Testing full model loading...", "MODEL_FULL")
        self.log("⚠️  This will use significant memory and time...")
        
        initial_memory = self.get_memory_usage()
        self.log(f"Initial memory: {initial_memory['rss_gb']}GB")
        
        try:
            start_time = time.time()
            
            # Use device_map="auto" for automatic device placement
            self.model = AutoModelForCausalLM.from_pretrained(
                "openai/gpt-oss-20b",
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            final_memory = self.get_memory_usage()
            memory_used = final_memory['rss_gb'] - initial_memory['rss_gb']
            
            self.results['model_full'] = {
                'loaded': True,
                'load_time': round(load_time, 2),
                'memory_used_gb': round(memory_used, 2),
                'total_memory_gb': final_memory['rss_gb'],
                'device': str(next(self.model.parameters()).device) if self.model else "unknown"
            }
            
            self.log(f"Model loaded: ✓ ({load_time:.2f}s, +{memory_used:.2f}GB memory)")
            self.log(f"Device: {self.results['model_full']['device']}")
            
            return True
            
        except Exception as e:
            final_memory = self.get_memory_usage()
            self.results['model_full'] = {
                'loaded': False,
                'error': str(e),
                'memory_at_failure_gb': final_memory['rss_gb']
            }
            self.log(f"Model loading failed: {e}")
            return False
    
    def test_simple_inference(self):
        """Test simple text generation"""
        if not self.model or not self.tokenizer:
            self.log("Skipping inference - model not loaded", "INFERENCE")
            return False
        
        self.log("Testing simple inference...", "INFERENCE")
        
        try:
            # Simple test prompt
            test_prompt = "The capital of France is"
            
            # Tokenize input
            inputs = self.tokenizer.encode(test_prompt, return_tensors="pt")
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            self.log(f"Input tokens: {inputs.shape[1]} on device: {device}")
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generation_time = time.time() - start_time
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_tokens = outputs.shape[1] - inputs.shape[1]
            tokens_per_second = new_tokens / generation_time if generation_time > 0 else 0
            
            self.results['inference'] = {
                'success': True,
                'generation_time': round(generation_time, 3),
                'input_tokens': inputs.shape[1],
                'output_tokens': new_tokens,
                'tokens_per_second': round(tokens_per_second, 2),
                'generated_text': generated_text,
                'prompt': test_prompt
            }
            
            self.log(f"Inference: ✓ ({generation_time:.3f}s, {tokens_per_second:.2f} tok/s)")
            self.log(f"Generated: '{generated_text}'")
            
            return True
            
        except Exception as e:
            self.results['inference'] = {
                'success': False,
                'error': str(e)
            }
            self.log(f"Inference failed: {e}")
            return False
    
    def test_pipeline_inference(self):
        """Test using pipeline for easier inference"""
        self.log("Testing pipeline inference...", "PIPELINE")
        
        try:
            # Create pipeline
            start_time = time.time()
            pipe = pipeline(
                "text-generation",
                model="openai/gpt-oss-20b",
                tokenizer="openai/gpt-oss-20b",
                torch_dtype=torch.float16,
                device_map="auto",
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True
                }
            )
            setup_time = time.time() - start_time
            
            self.log(f"Pipeline created: ✓ ({setup_time:.2f}s)")
            
            # Test generation
            test_prompt = "Explain quantum computing in simple terms:"
            
            start_time = time.time()
            results = pipe(
                test_prompt,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                return_full_text=False
            )
            generation_time = time.time() - start_time
            
            generated_text = results[0]['generated_text']
            
            # Estimate tokens per second
            estimated_tokens = len(generated_text.split())  # Rough estimate
            tokens_per_second = estimated_tokens / generation_time if generation_time > 0 else 0
            
            self.results['pipeline'] = {
                'success': True,
                'setup_time': round(setup_time, 2),
                'generation_time': round(generation_time, 3),
                'estimated_tokens': estimated_tokens,
                'tokens_per_second_est': round(tokens_per_second, 2),
                'generated_text': generated_text,
                'prompt': test_prompt
            }
            
            self.log(f"Pipeline inference: ✓ ({generation_time:.3f}s, ~{tokens_per_second:.2f} tok/s)")
            self.log(f"Generated: '{generated_text[:100]}...'")
            
            return True
            
        except Exception as e:
            self.results['pipeline'] = {
                'success': False,
                'error': str(e)
            }
            self.log(f"Pipeline inference failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up loaded models to free memory"""
        self.log("Cleaning up models...", "CLEANUP")
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear MPS cache if available
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        final_memory = self.get_memory_usage()
        self.log(f"Memory after cleanup: {final_memory['rss_gb']}GB")
    
    def run_all_tests(self, test_full_loading=False):
        """Run all inference tests"""
        self.log("Starting Model Inference Tests", "START")
        self.log("=" * 60)
        
        initial_memory = self.get_memory_usage()
        self.log(f"Initial memory usage: {initial_memory['rss_gb']}GB")
        
        # Test tokenizer
        tokenizer_ok = self.test_tokenizer_loading()
        
        # Test model config
        config_ok = self.test_model_loading_light()
        
        # Test full model loading (optional, memory intensive)
        if test_full_loading:
            model_ok = self.test_model_loading_full()
            
            if model_ok:
                # Test inference
                self.test_simple_inference()
        else:
            self.log("Skipping full model loading (use test_full_loading=True to enable)", "MODEL_FULL")
            
        # Test pipeline (alternative approach)
        # Note: This will also load the full model
        if test_full_loading:
            self.log("Testing pipeline approach...", "PIPELINE")
            self.test_pipeline_inference()
        
        # Cleanup
        self.cleanup()
        
        # Generate summary
        self.generate_inference_summary()
        
        return self.results
    
    def generate_inference_summary(self):
        """Generate inference test summary"""
        self.log("=" * 60)
        self.log("MODEL INFERENCE TEST SUMMARY", "SUMMARY")
        self.log("=" * 60)
        
        # Tokenizer
        tokenizer_status = "✓ SUCCESS" if self.results.get('tokenizer', {}).get('loaded', False) else "✗ FAILED"
        self.log(f"Tokenizer Loading: {tokenizer_status}")
        
        # Model Config
        config_status = "✓ SUCCESS" if self.results.get('model_config', {}).get('loaded', False) else "✗ FAILED"
        self.log(f"Model Config: {config_status}")
        
        # Full Model
        if 'model_full' in self.results:
            model_status = "✓ SUCCESS" if self.results['model_full'].get('loaded', False) else "✗ FAILED"
            self.log(f"Full Model Loading: {model_status}")
            
            if self.results['model_full'].get('loaded', False):
                self.log(f"Load Time: {self.results['model_full']['load_time']}s")
                self.log(f"Memory Used: {self.results['model_full']['memory_used_gb']}GB")
        
        # Inference
        if 'inference' in self.results:
            inference_status = "✓ SUCCESS" if self.results['inference'].get('success', False) else "✗ FAILED"
            self.log(f"Direct Inference: {inference_status}")
            
            if self.results['inference'].get('success', False):
                self.log(f"Speed: {self.results['inference']['tokens_per_second']} tokens/second")
        
        # Pipeline
        if 'pipeline' in self.results:
            pipeline_status = "✓ SUCCESS" if self.results['pipeline'].get('success', False) else "✗ FAILED"
            self.log(f"Pipeline Inference: {pipeline_status}")
            
            if self.results['pipeline'].get('success', False):
                self.log(f"Speed: ~{self.results['pipeline']['tokens_per_second_est']} tokens/second")
        
        self.log("=" * 60)

if __name__ == "__main__":
    import sys
    
    # Check if user wants full testing
    test_full = "--full" in sys.argv
    
    if test_full:
        print("⚠️  Running FULL tests including model loading (will use significant memory and time)")
        print("⚠️  Make sure you have sufficient RAM and time available")
        time.sleep(3)
    else:
        print("Running LIGHT tests (tokenizer + config only)")
        print("Use --full flag to test actual model loading and inference")
    
    inference_test = ModelInferenceTest()
    results = inference_test.run_all_tests(test_full_loading=test_full)
    
    # Save results
    filename = 'model_inference_full_results.json' if test_full else 'model_inference_light_results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nInference test results saved to: {filename}")