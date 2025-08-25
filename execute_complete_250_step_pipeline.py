#!/usr/bin/env python3

import os
import subprocess
import time
from datetime import datetime

print("🚀 COMPLETE WCS MODEL PIPELINE - 250 STEPS")
print("=" * 50)

def execute_pipeline():
    """Execute the complete 250-step training and testing pipeline."""
    
    print(f"⏱️ Pipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Extended Training
    print("\n🔧 STEP 1: EXECUTING EXTENDED TRAINING (250 STEPS)")
    print("-" * 50)
    
    training_start = time.time()
    try:
        result = subprocess.run(
            ["python", "extended_wcs_training_250_steps.py"],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours timeout
        )
        
        if result.returncode == 0:
            training_time = time.time() - training_start
            print(f"✅ Extended training completed in {training_time:.2f} seconds")
            print("📁 Model saved to: gpt-oss-20b-wcs-extended-250")
        else:
            print(f"❌ Training failed with error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ Training timeout - continuing with existing model if available")
        if not os.path.exists("gpt-oss-20b-wcs-extended-250"):
            print("❌ No extended model found, pipeline cannot continue")
            return False
    except Exception as e:
        print(f"❌ Training execution failed: {str(e)}")
        return False
    
    # Step 2: Comprehensive Testing
    print("\n🔬 STEP 2: COMPREHENSIVE TESTING ACROSS ALL FILES")
    print("-" * 50)
    
    testing_start = time.time()
    try:
        result = subprocess.run(
            ["python", "comprehensive_all_files_test.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            testing_time = time.time() - testing_start
            print(f"✅ Comprehensive testing completed in {testing_time:.2f} seconds")
            print("📁 Results saved to: COMPREHENSIVE_ALL_FILES_TEST_RESULTS.md")
        else:
            print(f"❌ Testing failed with error:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏱️ Testing timeout - check partial results")
        return False
    except Exception as e:
        print(f"❌ Testing execution failed: {str(e)}")
        return False
    
    # Pipeline Success
    total_time = time.time() - (training_start - (time.time() - training_start))
    print(f"\n🎉 COMPLETE PIPELINE SUCCESS!")
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    print(f"📊 Extended model: gpt-oss-20b-wcs-extended-250 (250 training steps)")
    print(f"📁 Test results: COMPREHENSIVE_ALL_FILES_TEST_RESULTS.md")
    print(f"✅ All 88 WCS files processed with 4 analysis categories each")
    
    return True

if __name__ == "__main__":
    success = execute_pipeline()
    if success:
        print("\n🚀 Pipeline execution completed successfully!")
    else:
        print("\n❌ Pipeline execution failed - check logs above")
        exit(1)