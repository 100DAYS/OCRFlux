#!/usr/bin/env python3
"""
Invoice Extraction with Quantized Models
Uses 4-bit quantized models for better accuracy with lower memory usage
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import torch

# OCRFlux imports
from vllm import LLM, SamplingParams
from ocrflux.inference import parse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_gpu_memory():
    """Check available GPU memory and provide recommendations."""
    if not torch.cuda.is_available():
        logger.error("No GPU detected! This pipeline requires a CUDA-capable GPU.")
        return False

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total GPU memory: {gpu_memory_gb:.1f} GB")

    if gpu_memory_gb < 12:
        logger.warning("Less than 12GB GPU memory. May need to reduce batch size.")
    else:
        logger.info("Sufficient GPU memory for OCRFlux + quantized 7B model")

    return True


def run_ocrflux(pdf_path: str, model_path: str = "ChatDOC/OCRFlux-3B",
                gpu_memory: float = 0.5, dtype: str = "auto") -> Optional[str]:
    """Process PDF through OCRFlux to generate markdown."""
    logger.info(f"Processing PDF with OCRFlux: {pdf_path}")

    try:
        # Initialize OCRFlux model
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory,
            max_model_len=8192,
            trust_remote_code=True,
            dtype=dtype
        )

        # Process the PDF
        result = parse(llm, pdf_path)

        if result is not None and 'document_text' in result:
            markdown_text = result['document_text']
            logger.info(f"Successfully converted PDF to markdown ({len(markdown_text)} characters)")

            # Save markdown
            markdown_path = Path(pdf_path).with_suffix('.md')
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            logger.info(f"Saved markdown to: {markdown_path}")

            return markdown_text
        else:
            logger.error("OCRFlux processing failed")
            return None

    except Exception as e:
        logger.error(f"Error during OCRFlux processing: {e}")
        return None


def extract_with_quantized_model(markdown_text: str, model_path: str,
                                 gpu_memory: float = 0.5) -> Optional[Dict[str, Any]]:
    """
    Extract invoice data using quantized model.

    Args:
        markdown_text: OCRFlux-generated markdown
        model_path: Path to quantized model (e.g., Qwen/Qwen2.5-7B-Instruct-AWQ)
        gpu_memory: GPU memory utilization

    Returns:
        Extracted invoice data dictionary
    """
    logger.info(f"Extracting with quantized model: {model_path}")

    # Enhanced invoice extraction prompt with serial numbers
    prompt = f"""You are a data extraction specialist. Extract invoice information from the following markdown text and return it as valid JSON.

Required fields:
{{
  "invoice_number": "string",
  "invoice_date": "string (YYYY-MM-DD)",
  "due_date": "string (YYYY-MM-DD)",
  "vendor": {{"name": "string", "address": "string", "tax_id": "string"}},
  "customer": {{"name": "string", "address": "string", "tax_id": "string"}},
  "line_items": [
    {{
      "position": "number (line number)",
      "article_number": "string (product code/SKU)",
      "description": "string",
      "quantity": "number",
      "unit_price": "number (price per unit)",
      "total": "number (line total)",
      "serial_numbers": ["array of serial number strings"]
    }}
  ],
  "subtotal": "number",
  "tax_rate": "number (percentage)",
  "tax_amount": "number",
  "total_amount": "number",
  "currency": "string",
  "payment_terms": "string"
}}

Instructions:
1. Extract ALL line items with complete details
2. Serial numbers may appear below product descriptions
3. Include all serial numbers as an array (empty array if none)
4. Parse numbers correctly (remove currency symbols)
5. Return ONLY valid JSON

Markdown Text:
{markdown_text}

JSON Output:"""

    try:
        # Initialize quantized model
        # vLLM automatically detects and handles quantization
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory,
            trust_remote_code=True,
            quantization="awq" if "AWQ" in model_path or "awq" in model_path else "gptq",
            max_model_len=8192  # Reduce if memory issues
        )

        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=2000
        )

        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # Clean response
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        # Parse JSON
        extracted_data = json.loads(response)
        logger.info("Successfully extracted invoice data")
        return extracted_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Raw response: {response[:1000]}...")
        # Try to extract JSON from response if it contains extra text
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group())
                logger.info("Successfully extracted JSON from response")
                return extracted_data
            except:
                pass
        return None
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract invoice data using quantized models for better accuracy"
    )
    parser.add_argument("pdf_path", help="Path to PDF invoice")
    parser.add_argument(
        "--ocrflux-model",
        default="ChatDOC/OCRFlux-3B",
        help="OCRFlux model (default: ChatDOC/OCRFlux-3B)"
    )
    parser.add_argument(
        "--extract-model",
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="Quantized extraction model (default: Qwen2.5-7B-AWQ)"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.5,
        help="GPU memory utilization (0-1, default: 0.5 for sequential model loading)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save extracted data as JSON"
    )

    args = parser.parse_args()

    # Check GPU
    if not check_gpu_memory():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Validate PDF
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF not found: {args.pdf_path}")
        sys.exit(1)

    # Step 1: OCRFlux
    markdown_text = run_ocrflux(
        args.pdf_path,
        args.ocrflux_model,
        args.gpu_memory
    )

    if not markdown_text:
        logger.error("Failed to generate markdown")
        sys.exit(1)

    # Step 2: Extract with quantized model
    extracted_data = extract_with_quantized_model(
        markdown_text,
        args.extract_model,
        args.gpu_memory
    )

    if not extracted_data:
        logger.error("Failed to extract data")
        sys.exit(1)

    # Display results
    print("\n" + "="*60)
    print("EXTRACTED INVOICE DATA (Quantized Model)")
    print("="*60)
    print(json.dumps(extracted_data, indent=2, ensure_ascii=False))

    # Save JSON if requested
    if args.save_json:
        json_path = Path(args.pdf_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to: {json_path}")


if __name__ == "__main__":
    main()