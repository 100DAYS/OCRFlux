#!/usr/bin/env python3
"""
Simplified Invoice Extraction Script
Uses OCRFlux for OCR and a single model for extraction
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from vllm import LLM, SamplingParams
from ocrflux.inference import parse


def check_gpu_memory():
    """Quick GPU memory check."""
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected! This requires a CUDA-capable GPU.")
        return False

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f} GB)")

    if gpu_memory_gb < 12:
        print("WARNING: Less than 12GB GPU memory. OCRFlux may fail.")
        return False
    return True


def process_invoice(pdf_path: str, ocrflux_model: str = "ChatDOC/OCRFlux-3B", gpu_memory: float = 0.8, dtype: str = "auto"):
    """
    Complete pipeline: PDF -> OCRFlux -> Data Extraction -> Print Results
    """
    print(f"Processing: {pdf_path}\n")

    # Step 1: OCRFlux - Convert PDF to Markdown
    print("Step 1: Converting PDF to Markdown with OCRFlux...")
    llm_ocr = LLM(
        model=ocrflux_model,
        gpu_memory_utilization=gpu_memory,
        max_model_len=8192,
        trust_remote_code=True,
        dtype=dtype  # Support for older GPUs like V100
    )

    result = parse(llm_ocr, pdf_path)
    if result is None or 'document_text' not in result:
        print("ERROR: Failed to convert PDF to markdown")
        sys.exit(1)

    markdown_text = result['document_text']
    print(f"✓ Generated {len(markdown_text)} characters of markdown\n")

    # Save markdown for reference
    md_path = Path(pdf_path).with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write(markdown_text)

    # Step 2: Extract Invoice Data
    print("Step 2: Extracting invoice data...")

    # For extraction, we'll use the same Qwen model with a text prompt
    # Note: This is a workaround - ideally use a text-only model
    extraction_prompt = f"""Extract the following invoice information and return as JSON:

Fields to extract:
- invoice_number
- invoice_date (format: YYYY-MM-DD)
- vendor_name
- vendor_address
- customer_name
- customer_address
- line_items (array with: description, quantity, unit_price, total)
- subtotal
- tax_amount
- total_amount
- currency

From this text:
{markdown_text[:8000]}  # Limit context length

Return only valid JSON, no explanation:"""

    # Since we need an image for Qwen2.5-VL, create a dummy one
    from PIL import Image
    dummy_image = Image.new('RGB', (28, 28), color='black')

    # Build query in Qwen format
    qwen_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{extraction_prompt}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2000,
        stop=["<|im_end|>", "```"]
    )

    # Generate with dummy image
    outputs = llm_ocr.generate(
        prompts=[qwen_prompt],
        sampling_params=sampling_params,
        use_tqdm=False,
        multi_modal_data={"image": [dummy_image]}
    )

    response = outputs[0].outputs[0].text.strip()

    # Clean response
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]

    try:
        invoice_data = json.loads(response)
        print("✓ Successfully extracted invoice data\n")
    except json.JSONDecodeError:
        print(f"ERROR: Failed to parse JSON response:\n{response[:500]}")
        sys.exit(1)

    # Step 3: Display Results
    print("="*60)
    print("EXTRACTED INVOICE DATA")
    print("="*60)
    print(f"\nInvoice #: {invoice_data.get('invoice_number', 'N/A')}")
    print(f"Date: {invoice_data.get('invoice_date', 'N/A')}")
    print(f"\nVendor: {invoice_data.get('vendor_name', 'N/A')}")
    print(f"Customer: {invoice_data.get('customer_name', 'N/A')}")

    if line_items := invoice_data.get('line_items'):
        print(f"\nLine Items ({len(line_items)}):")
        for item in line_items:
            print(f"  - {item.get('description', 'N/A')}: {item.get('total', 'N/A')}")

    print(f"\nTotal: {invoice_data.get('total_amount', 'N/A')} {invoice_data.get('currency', '')}")
    print("="*60)

    # Save JSON
    json_path = Path(pdf_path).with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(invoice_data, f, indent=2)
    print(f"\n✓ Saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract invoice data from PDF")
    parser.add_argument("pdf_path", help="Path to PDF invoice")
    parser.add_argument("--ocrflux-model", default="ChatDOC/OCRFlux-3B", help="OCRFlux model path/ID (default: ChatDOC/OCRFlux-3B)")
    parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU memory utilization (0-1, default: 0.8)")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
                       help="Model dtype (use float32 for V100 GPUs)")
    args = parser.parse_args()

    # Check GPU memory
    if not check_gpu_memory():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    if not Path(args.pdf_path).exists():
        print(f"ERROR: File not found: {args.pdf_path}")
        sys.exit(1)

    process_invoice(args.pdf_path, args.ocrflux_model, args.gpu_memory, args.dtype)