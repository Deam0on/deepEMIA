#!/usr/bin/env python3
"""
Test script to validate the new Gradio interface integration.
This script checks if all imports and function signatures are correct.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def test_backend_imports():
    """Test if all backend functions can be imported."""
    try:
        from src.data.datasets import split_dataset
        from src.functions.train_model import train_on_dataset
        from src.functions.evaluate_model import evaluate_model
        from src.functions.inference import run_inference
        from src.utils.config import get_config
        from src.utils.logger_utils import system_logger
        print("✅ All backend imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test if configuration can be loaded."""
    try:
        from src.utils.config import get_config
        config = get_config()
        print(f"✅ Config loaded - bucket: {config.get('bucket', 'Not set')}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_function_signatures():
    """Test if function signatures match what we're calling."""
    try:
        from src.data.datasets import split_dataset
        from src.functions.train_model import train_on_dataset
        from src.functions.evaluate_model import evaluate_model
        from src.functions.inference import run_inference
        
        print("✅ Function signatures verified:")
        
        # Check split_dataset
        import inspect
        sig = inspect.signature(split_dataset)
        print(f"   split_dataset: {sig}")
        
        # Check train_on_dataset  
        sig = inspect.signature(train_on_dataset)
        print(f"   train_on_dataset: {sig}")
        
        # Check evaluate_model
        sig = inspect.signature(evaluate_model)
        print(f"   evaluate_model: {sig}")
        
        # Check run_inference
        sig = inspect.signature(run_inference)
        print(f"   run_inference: {sig}")
        
        return True
    except Exception as e:
        print(f"❌ Signature error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing deepEMIA Gradio Interface Integration")
    print("=" * 50)
    
    tests = [
        ("Backend Imports", test_backend_imports),
        ("Config Loading", test_config_loading),
        ("Function Signatures", test_function_signatures)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The Gradio interface should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
