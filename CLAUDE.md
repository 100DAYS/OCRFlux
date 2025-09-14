# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCRFlux is a multimodal large language model toolkit for converting PDFs and images into clean, readable, plain Markdown text. It uses a 3B parameter visual language model (VLM) based on Qwen2.5-VL architecture.

## Development Commands

### Installation
```bash
# Create and activate conda environment
conda create -n ocrflux python=3.11
conda activate ocrflux

# Install dependencies
pip install -e . --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
```

### Core Pipeline Operations
```bash
# Process a single PDF
python -m ocrflux.pipeline ./localworkspace --data test.pdf --model /path/to/OCRFlux-3B

# Process an image
python -m ocrflux.pipeline ./localworkspace --data test_page.png --model /path/to/OCRFlux-3B

# Process directory of files
python -m ocrflux.pipeline ./localworkspace --data test_pdf_dir/* --model /path/to/OCRFlux-3B

# Convert JSONL results to Markdown
python -m ocrflux.jsonl_to_markdown ./localworkspace
```

### Server Operations
```bash
# Start vLLM server
bash ocrflux/server.sh /path/to/model 30024

# Or directly with vllm
vllm serve /path/to/model --port 30024 --max-model-len 8192 --gpu_memory_utilization 0.8
```

### Evaluation Tasks
```bash
# Page to Markdown evaluation
python -m ocrflux.pipeline ./eval_result --task pdf2markdown --data /data/pdfs/*.pdf --model /path/to/model
python -m eval.eval_page_to_markdown ./eval_result --gt_file /data/data.jsonl

# Element merge detection
python -m eval.gen_element_merge_detect_data /data/OCRFlux-bench-cross
python -m ocrflux.pipeline ./eval_result --task merge_pages --data /data/jsons/*.json --model /path/to/model
python -m eval.eval_element_merge_detect ./eval_result --gt_file /data/data.jsonl

# Table to HTML evaluation
python -m ocrflux.pipeline ./eval_result --task pdf2markdown --data /data/images/*.png --model /path/to/model
python -m eval.eval_table_to_html ./eval_result --gt_file /data/data.jsonl

# HTML table merge
python -m eval.gen_html_table_merge_data /data/OCRFlux-pubtabnet-cross
python -m ocrflux.pipeline ./eval_result --task merge_tables --data /data/jsons/*.json --model /path/to/model
python -m eval.eval_html_table_merge ./eval_result --gt_file /data/data.jsonl
```

## Architecture Overview

### Core Components

**ocrflux/pipeline.py**: Main entry point for batch processing PDFs and images. Implements three task types:
- `pdf2markdown`: Converts PDF pages or images to Markdown
- `merge_pages`: Detects elements that need merging across consecutive pages
- `merge_tables`: Merges split table fragments

**ocrflux/inference.py**: Provides offline inference API using vLLM. Key function `parse()` processes files and returns structured results with document text and per-page outputs.

**ocrflux/client.py**: Client for online inference via vLLM server. Implements asynchronous request processing with retry logic and fallback handling.

**ocrflux/prompts.py**: Contains prompt templates for different OCR tasks:
- Page-to-Markdown conversion prompts
- Element merge detection prompts
- HTML table merge prompts

**ocrflux/work_queue.py**: Implements work queue system for parallel processing with LocalWorkQueue for file-based task management.

**ocrflux/image_utils.py**: Handles PDF rendering and image preprocessing using pypdfium2.

**ocrflux/table_format.py**: Utilities for converting table matrices to HTML format.

**ocrflux/metrics.py**: Tracks performance metrics and worker statistics during batch processing.

### Processing Flow

1. **Input Processing**: PDF/image files are rendered to images at specified resolution (default 1024px longest side)
2. **VLM Inference**: Images are processed through Qwen2.5-VL model using vLLM for efficient batch inference
3. **Cross-Page Merging**: Consecutive pages analyzed for split tables/paragraphs that need merging
4. **Output Generation**: Results saved as JSONL with document text, per-page texts, and fallback pages

### Key Configuration Parameters

- `--gpu_memory_utilization`: GPU memory usage (default 0.8)
- `--tensor_parallel_size`: For multi-GPU setups
- `--max_page_retries`: Number of retries for failed pages
- `--skip_cross_page_merge`: Option to disable cross-page merging
- `--dtype`: Model precision (auto, float16, bfloat16, float32)

## Model Requirements

- Minimum 12GB GPU memory (24GB+ recommended)
- Supports NVIDIA GPUs: RTX 3090, 4090, L40S, A100, H100
- For V100 GPUs without bf16 support, use `--dtype float32`
- Multi-GPU support via tensor parallelism for smaller GPUs

## Invoice Data Extraction Pipeline

### Overview
Extended OCRFlux with structured data extraction capabilities for invoices using quantized LLMs.

### Tested Configuration (2024-09-14)
Successfully tested on NVIDIA RTX 4000 SFF Ada (19.5GB) with German invoice PDF.

#### Models Used:
1. **OCRFlux-3B**: PDF to Markdown conversion
2. **Qwen2.5-7B-Instruct-GPTQ-Int4**: 4-bit quantized model for extraction

#### Test Commands:
```bash
# Full precision 3B model (baseline)
python extract_invoice_data.py invoice.pdf \
    --extract-model Qwen/Qwen2.5-3B-Instruct \
    --gpu-memory 0.7 \
    --save-json

# 4-bit quantized 7B model (recommended)
python extract_invoice_quantized.py invoice.pdf \
    --extract-model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --gpu-memory 0.5 \
    --save-json
```

#### Results:
- **Memory Usage**: GPTQ 7B (5.2GB) < Full 3B (5.79GB)
- **Speed**: GPTQ 7B 2.4x faster on input processing
- **Accuracy**: Both models correctly extracted all data including serial numbers
- **Recommendation**: Use GPTQ-Int4 quantized models for production

### Key Features:
- Extracts line items with serial numbers
- Handles multiple currencies and tax rates
- Supports multi-language invoices (tested with German)
- Automatic model downloading from HuggingFace

## ⚠️ Important: Data Privacy for Public Repository

**This is a PUBLIC repository.** When testing or documenting extraction features:
- **Always anonymize sensitive data** before committing (serial numbers, prices, customer names)
- **Replace real values** with generic examples
- **Do not commit** actual invoice files or customer data
- **Test data** should use synthetic or properly anonymized information

All examples in this documentation have been anonymized for privacy.