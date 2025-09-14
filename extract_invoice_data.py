#!/usr/bin/env python3
"""
Invoice Data Extraction Pipeline
Processes PDF -> OCRFlux Markdown -> Structured Data Extraction
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
        logger.warning("Less than 12GB GPU memory detected. OCRFlux may fail to load.")
        logger.warning("Consider using --gpu-memory 0.5 or lower")
        return False
    elif gpu_memory_gb < 24:
        logger.info("12-24GB GPU memory detected. Can run OCRFlux + small text model.")
        logger.info("Recommended: Use Qwen2.5-3B or Llama-3.2-3B for extraction")
    else:
        logger.info("24GB+ GPU memory detected. Can run OCRFlux + larger models.")
        logger.info("Recommended: Use Qwen2.5-7B for best extraction accuracy")

    return True


def run_ocrflux(pdf_path: str, model_path: str, gpu_memory: float = 0.8, dtype: str = "auto") -> Optional[str]:
    """
    Process PDF through OCRFlux to generate markdown.

    Args:
        pdf_path: Path to the PDF file
        model_path: Path to OCRFlux model
        gpu_memory: GPU memory utilization (0-1)

    Returns:
        Markdown text or None if processing fails
    """
    logger.info(f"Processing PDF with OCRFlux: {pdf_path}")

    # Ensure model is downloaded
    model_path = ensure_model_downloaded(model_path)

    try:
        # Initialize OCRFlux model
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory,
            max_model_len=8192,
            trust_remote_code=True,
            dtype=dtype  # Support for older GPUs
        )

        # Process the PDF
        result = parse(llm, pdf_path)

        if result is not None and 'document_text' in result:
            markdown_text = result['document_text']
            logger.info(f"Successfully converted PDF to markdown ({len(markdown_text)} characters)")

            # Optionally save markdown to file
            markdown_path = Path(pdf_path).with_suffix('.md')
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            logger.info(f"Saved markdown to: {markdown_path}")

            return markdown_text
        else:
            logger.error("OCRFlux processing failed or returned no content")
            return None

    except Exception as e:
        logger.error(f"Error during OCRFlux processing: {e}")
        return None


def ensure_model_downloaded(model_path: str) -> str:
    """
    Ensure model is downloaded, download if necessary.

    Args:
        model_path: HuggingFace model ID or local path

    Returns:
        Local path to model
    """
    # Check if it's a local path that exists
    if os.path.exists(model_path):
        return model_path

    # Try to download from HuggingFace
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading model: {model_path}")
        local_dir = f"./models/{model_path.replace('/', '_')}"
        snapshot_download(repo_id=model_path, local_dir=local_dir)
        logger.info(f"Model downloaded to: {local_dir}")
        return local_dir
    except ImportError:
        logger.warning("huggingface_hub not installed, assuming model path is correct")
        return model_path
    except Exception as e:
        logger.warning(f"Could not download model: {e}, attempting to use as-is")
        return model_path


def extract_invoice_data(markdown_text: str, model_path: str, gpu_memory: float = 0.8, dtype: str = "auto") -> Optional[Dict[str, Any]]:
    """
    Extract structured invoice data from markdown using LLM.

    Args:
        markdown_text: OCRFlux-generated markdown
        model_path: Path to text extraction model (e.g., Qwen2.5-7B-Instruct)
        gpu_memory: GPU memory utilization (0-1)

    Returns:
        Dictionary with extracted invoice data or None if extraction fails
    """
    logger.info("Extracting invoice data from markdown")

    # Ensure model is downloaded
    model_path = ensure_model_downloaded(model_path)

    # Define the extraction schema with enhanced line items
    schema = {
        "invoice_number": "string",
        "invoice_date": "string (YYYY-MM-DD format if possible)",
        "due_date": "string (YYYY-MM-DD format if possible)",
        "vendor": {
            "name": "string",
            "address": "string",
            "tax_id": "string (if available)"
        },
        "customer": {
            "name": "string",
            "address": "string",
            "tax_id": "string (if available)"
        },
        "line_items": [
            {
                "position": "number (item position/line number)",
                "article_number": "string (product/article code)",
                "description": "string (product description)",
                "quantity": "number (amount ordered)",
                "unit_price": "number (price per single unit)",
                "total": "number (quantity * unit_price)",
                "serial_numbers": "array of strings (serial numbers if available, empty array if none)"
            }
        ],
        "subtotal": "number",
        "tax_rate": "number (percentage)",
        "tax_amount": "number",
        "total_amount": "number",
        "currency": "string (e.g., USD, EUR)",
        "payment_terms": "string",
        "payment_method": "string (if specified)",
        "notes": "string (any additional notes or terms)"
    }

    # Build the extraction prompt
    prompt = f"""You are a data extraction specialist. Extract invoice information from the following markdown text and return it as valid JSON.

Required Schema:
{json.dumps(schema, indent=2)}

Instructions:
1. Extract ALL line items as an array, even if there's only one
2. For each line item, extract:
   - position: The line/position number if shown
   - article_number: Product code/SKU/article number
   - description: Full product description
   - quantity: Number of units (default to 1 if not specified)
   - unit_price: Price per single unit
   - total: Line total (quantity * unit_price)
   - serial_numbers: Array of serial numbers. If multiple serial numbers are listed, include all. If none, use empty array []
3. Serial numbers may appear on separate lines below the product description
4. For missing fields, use null
5. Ensure numbers are parsed correctly (remove currency symbols and thousand separators)
6. Parse dates to YYYY-MM-DD format when possible
7. Return ONLY valid JSON, no explanations or markdown formatting

Markdown Text:
{markdown_text}

JSON Output:"""

    try:
        # Initialize text LLM for extraction
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory,
            trust_remote_code=True,
            dtype=dtype  # Support for older GPUs
        )

        # Set sampling parameters for deterministic output
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1,
            max_tokens=2000,
            stop=["```", "\n\n\n"]
        )

        # Generate extraction
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        # Clean up response if needed
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
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {response[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Error during data extraction: {e}")
        return None


def print_invoice_data(data: Dict[str, Any]) -> None:
    """
    Pretty print extracted invoice data.

    Args:
        data: Dictionary containing extracted invoice data
    """
    print("\n" + "="*60)
    print("EXTRACTED INVOICE DATA")
    print("="*60)

    # Basic Information
    print(f"\nInvoice Number: {data.get('invoice_number', 'N/A')}")
    print(f"Invoice Date: {data.get('invoice_date', 'N/A')}")
    print(f"Due Date: {data.get('due_date', 'N/A')}")
    print(f"Currency: {data.get('currency', 'N/A')}")

    # Vendor Information
    if vendor := data.get('vendor'):
        print(f"\nVendor:")
        print(f"  Name: {vendor.get('name', 'N/A')}")
        print(f"  Address: {vendor.get('address', 'N/A')}")
        if tax_id := vendor.get('tax_id'):
            print(f"  Tax ID: {tax_id}")

    # Customer Information
    if customer := data.get('customer'):
        print(f"\nCustomer:")
        print(f"  Name: {customer.get('name', 'N/A')}")
        print(f"  Address: {customer.get('address', 'N/A')}")
        if tax_id := customer.get('tax_id'):
            print(f"  Tax ID: {tax_id}")

    # Line Items
    if line_items := data.get('line_items'):
        print(f"\nLine Items ({len(line_items)} items):")
        print("-" * 70)
        for item in line_items:
            pos = item.get('position', 'N/A')
            art_nr = item.get('article_number', 'N/A')
            desc = item.get('description', 'N/A')
            qty = item.get('quantity', 'N/A')
            price = item.get('unit_price', 'N/A')
            total = item.get('total', 'N/A')
            serials = item.get('serial_numbers', [])

            print(f"  Pos {pos}: [{art_nr}] {desc}")
            print(f"         Qty: {qty} x ${price} = ${total}")
            if serials:
                print(f"         Serial Numbers: {', '.join(serials)}")
        print("-" * 70)

    # Totals
    print(f"\nFinancial Summary:")
    print(f"  Subtotal: ${data.get('subtotal', 'N/A')}")
    if tax_rate := data.get('tax_rate'):
        print(f"  Tax Rate: {tax_rate}%")
    print(f"  Tax Amount: ${data.get('tax_amount', 'N/A')}")
    print(f"  Total Amount: ${data.get('total_amount', 'N/A')}")

    # Payment Information
    if payment_terms := data.get('payment_terms'):
        print(f"\nPayment Terms: {payment_terms}")
    if payment_method := data.get('payment_method'):
        print(f"Payment Method: {payment_method}")

    # Additional Notes
    if notes := data.get('notes'):
        print(f"\nNotes: {notes}")

    print("\n" + "="*60)


def save_json_output(data: Dict[str, Any], pdf_path: str) -> None:
    """
    Save extracted data as JSON file.

    Args:
        data: Extracted invoice data
        pdf_path: Original PDF path (used to generate output filename)
    """
    json_path = Path(pdf_path).with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved extracted data to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured invoice data from PDF using OCRFlux and LLM"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF invoice file"
    )
    parser.add_argument(
        "--ocrflux-model",
        default="ChatDOC/OCRFlux-3B",
        help="Path/ID of OCRFlux model (default: ChatDOC/OCRFlux-3B)"
    )
    parser.add_argument(
        "--extract-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Path/ID of extraction LLM (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.8,
        help="GPU memory utilization (0-1, default: 0.8)"
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype (use float32 for V100 GPUs, default: auto)"
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save extracted data as JSON file"
    )
    parser.add_argument(
        "--markdown-only",
        action="store_true",
        help="Only generate markdown, skip extraction"
    )

    args = parser.parse_args()

    # Check GPU memory
    if not check_gpu_memory():
        response = input("\nGPU memory may be insufficient. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Validate PDF exists
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Step 1: Generate markdown with OCRFlux
    markdown_text = run_ocrflux(
        args.pdf_path,
        args.ocrflux_model,
        args.gpu_memory,
        args.dtype
    )

    if markdown_text is None:
        logger.error("Failed to generate markdown from PDF")
        sys.exit(1)

    if args.markdown_only:
        print("\nMarkdown generation complete. Skipping extraction.")
        sys.exit(0)

    # Step 2: Extract structured data
    extracted_data = extract_invoice_data(
        markdown_text,
        args.extract_model,
        args.gpu_memory,
        args.dtype
    )

    if extracted_data is None:
        logger.error("Failed to extract invoice data")
        sys.exit(1)

    # Step 3: Output results
    print_invoice_data(extracted_data)

    if args.save_json:
        save_json_output(extracted_data, args.pdf_path)

    logger.info("Invoice data extraction complete!")


if __name__ == "__main__":
    main()