#!/usr/bin/env python3
"""
Test if vLLM is using FlashInfer/Flash Attention
"""

import os
import sys

# Activate verbose logging to see backend selection
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

from vllm import LLM, SamplingParams

print("Initializing vLLM with OCRFlux model...")
print("Watch for backend selection messages...\n")

try:
    # Initialize with enough memory
    llm = LLM(
        model="ChatDOC/OCRFlux-3B",
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        trust_remote_code=True
    )

    print("\n✓ vLLM initialized successfully")
    print("Check the debug output above for attention backend selection")

    # Test inference
    prompt = "Hello"
    sampling_params = SamplingParams(temperature=0, max_tokens=10)

    print("\nTesting inference...")
    outputs = llm.generate([prompt], sampling_params)
    print("✓ Inference successful")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)