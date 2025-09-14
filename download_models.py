#!/usr/bin/env python3
"""
Model Download Utility for Invoice Extraction Pipeline
Downloads required models from HuggingFace Hub
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed")
    print("Please run: pip install huggingface-hub tqdm")
    sys.exit(1)


# Recommended models for invoice extraction
RECOMMENDED_MODELS = {
    "ocrflux": {
        "id": "ChatDOC/OCRFlux-3B",
        "description": "OCRFlux 3B model for PDF/image to Markdown conversion",
        "size": "~6GB"
    },
    "qwen-7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B Instruct - Best for accurate extraction",
        "size": "~15GB"
    },
    "qwen-3b": {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Qwen 2.5 3B Instruct - Lightweight alternative",
        "size": "~6GB"
    },
    "llama-3b": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Llama 3.2 3B - Efficient extraction model",
        "size": "~6GB"
    },
    "nuextract": {
        "id": "numind/NuExtract",
        "description": "NuExtract - Purpose-built for data extraction",
        "size": "~7GB"
    }
}


def download_model(model_id: str, local_dir: str = None) -> str:
    """
    Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model repository ID
        local_dir: Local directory to save model (optional)

    Returns:
        Path to downloaded model
    """
    if local_dir is None:
        # Default to ./models/model_name
        model_name = model_id.replace("/", "_")
        local_dir = f"./models/{model_name}"

    local_dir = Path(local_dir).absolute()

    # Check if already downloaded
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"✓ Model already exists at: {local_dir}")
        response = input("Re-download? (y/n): ").lower()
        if response != 'y':
            return str(local_dir)

    print(f"\nDownloading: {model_id}")
    print(f"Destination: {local_dir}")
    print("This may take several minutes depending on model size and connection speed...\n")

    try:
        # Create progress callback
        downloaded_files = []

        def progress_callback(filename):
            downloaded_files.append(filename)
            print(f"  Downloaded: {filename}")

        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            max_workers=4  # Parallel downloads
        )

        print(f"\n✓ Successfully downloaded to: {local_dir}")
        return str(local_dir)

    except Exception as e:
        print(f"\n✗ Error downloading {model_id}: {e}")
        return None


def list_models():
    """Display available recommended models."""
    print("\n" + "="*70)
    print("RECOMMENDED MODELS FOR INVOICE EXTRACTION")
    print("="*70)

    print("\nOCR Model (Required):")
    print("-"*40)
    model = RECOMMENDED_MODELS["ocrflux"]
    print(f"  {model['id']}")
    print(f"  {model['description']}")
    print(f"  Size: {model['size']}\n")

    print("\nExtraction Models (Choose One):")
    print("-"*40)
    for key in ["qwen-7b", "qwen-3b", "llama-3b", "nuextract"]:
        model = RECOMMENDED_MODELS[key]
        print(f"  {model['id']}")
        print(f"  {model['description']}")
        print(f"  Size: {model['size']}\n")

    print("="*70)


def interactive_download():
    """Interactive model download wizard."""
    print("\n" + "="*70)
    print("INVOICE EXTRACTION MODEL SETUP")
    print("="*70)

    models_to_download = []

    # Check OCRFlux
    print("\n1. OCRFlux Model (Required for PDF/Image OCR)")
    print("-"*40)
    ocrflux_path = "./models/ChatDOC_OCRFlux-3B"
    if Path(ocrflux_path).exists() and any(Path(ocrflux_path).iterdir()):
        print("✓ OCRFlux-3B already downloaded")
    else:
        print("✗ OCRFlux-3B not found")
        response = input("Download OCRFlux-3B? (y/n): ").lower()
        if response == 'y':
            models_to_download.append(("ChatDOC/OCRFlux-3B", ocrflux_path))

    # Choose extraction model
    print("\n2. Text Extraction Model (For structured data extraction)")
    print("-"*40)
    print("Options:")
    print("  1. Qwen2.5-7B-Instruct (~15GB) - Best accuracy")
    print("  2. Qwen2.5-3B-Instruct (~6GB) - Good balance")
    print("  3. Llama-3.2-3B-Instruct (~6GB) - Fast & efficient")
    print("  4. NuExtract (~7GB) - Purpose-built for extraction")
    print("  5. Skip (use OCRFlux only - not recommended)")

    choice = input("\nSelect extraction model (1-5): ").strip()

    model_map = {
        "1": ("Qwen/Qwen2.5-7B-Instruct", "./models/Qwen_Qwen2.5-7B-Instruct"),
        "2": ("Qwen/Qwen2.5-3B-Instruct", "./models/Qwen_Qwen2.5-3B-Instruct"),
        "3": ("meta-llama/Llama-3.2-3B-Instruct", "./models/meta-llama_Llama-3.2-3B-Instruct"),
        "4": ("numind/NuExtract", "./models/numind_NuExtract")
    }

    if choice in model_map:
        model_id, local_path = model_map[choice]
        if Path(local_path).exists() and any(Path(local_path).iterdir()):
            print(f"✓ {model_id} already downloaded")
        else:
            models_to_download.append((model_id, local_path))

    # Download selected models
    if models_to_download:
        print(f"\n{len(models_to_download)} model(s) will be downloaded:")
        for model_id, _ in models_to_download:
            print(f"  - {model_id}")

        response = input("\nProceed with download? (y/n): ").lower()
        if response == 'y':
            for model_id, local_path in models_to_download:
                result = download_model(model_id, local_path)
                if result is None:
                    print(f"Failed to download {model_id}")
        else:
            print("Download cancelled")
    else:
        print("\n✓ All required models are already downloaded")

    # Show usage example
    print("\n" + "="*70)
    print("USAGE EXAMPLE")
    print("="*70)
    print("\n# Extract invoice data from PDF:")
    print("python extract_invoice_data.py invoice.pdf \\")
    print("    --ocrflux-model ./models/ChatDOC_OCRFlux-3B \\")

    if choice in model_map:
        _, local_path = model_map[choice]
        print(f"    --extract-model {local_path} \\")

    print("    --save-json")
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Download models for invoice extraction pipeline"
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model ID to download (e.g., 'ChatDOC/OCRFlux-3B')"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recommended models"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save model (default: ./models/MODEL_NAME)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all recommended models"
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        sys.exit(0)

    if args.all:
        print("Downloading all recommended models...")
        for key, model_info in RECOMMENDED_MODELS.items():
            model_id = model_info["id"]
            print(f"\nDownloading: {model_id}")
            download_model(model_id)
        sys.exit(0)

    if args.model:
        # Direct model download
        result = download_model(args.model, args.output_dir)
        if result:
            print(f"\nModel ready at: {result}")
            print("\nUsage:")
            print(f"python extract_invoice_data.py invoice.pdf --ocrflux-model {result}")
        sys.exit(0 if result else 1)

    # Interactive mode
    interactive_download()


if __name__ == "__main__":
    main()