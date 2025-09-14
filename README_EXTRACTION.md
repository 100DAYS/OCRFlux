# Invoice Data Extraction Scripts

This directory contains scripts for extracting structured invoice data from PDFs using OCRFlux and LLMs.

## Tested Configuration (2024-09-14)

Successfully tested extraction pipeline on NVIDIA RTX 4000 SFF Ada (19.5GB) with real German invoice.

### Test File
- **Invoice Type**: German technology equipment invoice
- **Language**: German
- **Features Tested**: Multi-line items, serial numbers, European VAT

### Test Results

#### 1. Full Precision 3B Model
```bash
python extract_invoice_data.py invoice.pdf \
    --extract-model Qwen/Qwen2.5-3B-Instruct \
    --gpu-memory 0.7 \
    --save-json
```
- **Memory**: 5.79GB
- **Speed**: 78.6 tokens/s input, 39.1 tokens/s output
- **Result**: ✅ All data extracted including serial numbers

#### 2. 4-bit Quantized 7B Model (RECOMMENDED)
```bash
python extract_invoice_quantized.py invoice.pdf \
    --extract-model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --gpu-memory 0.5 \
    --save-json
```
- **Memory**: 5.2GB (less than 3B!)
- **Speed**: 189.6 tokens/s input, 49.6 tokens/s output (2.4x faster)
- **Result**: ✅ All data extracted including serial numbers

### Extracted Data Example
```json
{
  "line_items": [{
    "position": 1,
    "article_number": "12345678",
    "description": "Professional Camera Equipment Set",
    "quantity": 1,
    "unit_price": 2499.99,
    "total": 2499.99,
    "serial_numbers": ["SN-2024-0001", "SN-2024-0002", "SN-2024-0003"]
  }]
}
```

## Scripts

### 1. `extract_invoice_data.py` (Full Pipeline)
Complete extraction pipeline with separate models for OCR and extraction.

**Features:**
- Uses OCRFlux for PDF to Markdown conversion
- Supports separate text LLM for extraction (e.g., Qwen2.5-7B-Instruct)
- Comprehensive invoice schema extraction
- JSON output with pretty printing

**Usage:**
```bash
# Basic usage
python extract_invoice_data.py invoice.pdf

# With custom models
python extract_invoice_data.py invoice.pdf \
    --ocrflux-model ChatDOC/OCRFlux-3B \
    --extract-model Qwen/Qwen2.5-7B-Instruct \
    --save-json

# Generate only markdown (skip extraction)
python extract_invoice_data.py invoice.pdf --markdown-only
```

### 2. `extract_invoice_simple.py` (Simplified)
Simpler script using only OCRFlux model for both OCR and extraction.

**Features:**
- Single model for entire pipeline
- Faster processing (no model switching)
- Basic invoice field extraction
- Minimal dependencies

**Usage:**
```bash
# Basic usage (uses default ChatDOC/OCRFlux-3B)
python extract_invoice_simple.py invoice.pdf

# With custom model path
python extract_invoice_simple.py invoice.pdf --ocrflux-model /path/to/OCRFlux-3B
```

## Extracted Fields

Both scripts extract the following invoice information:
- Invoice number and date
- Vendor details (name, address, tax ID)
- Customer details (name, address, tax ID)
- Line items (description, quantity, unit price, total)
- Financial summary (subtotal, tax, total)
- Payment terms and methods
- Currency

## Output Files

For `invoice.pdf`, the scripts generate:
- `invoice.md` - OCRFlux markdown output
- `invoice.json` - Extracted structured data

## System Requirements

### GPU Memory Requirements:
- **Minimum**: 12GB VRAM (RTX 3060, RTX 3090, etc.)
  - Can run OCRFlux + small text model (3B params)
  - Use `--gpu-memory 0.6` if running into memory issues

- **Recommended**: 24GB+ VRAM (RTX 3090, RTX 4090, A100, etc.)
  - Can run OCRFlux + larger models (7B params)
  - Better extraction accuracy with larger models

- **For V100 GPUs** (no bfloat16 support):
  - Use `--dtype float32` flag
  - Requires more memory (~1.5x)

### Supported GPUs:
- NVIDIA RTX 3090/4090 (24GB) - Optimal
- NVIDIA RTX 3060/3070/3080 (12GB) - Works with small models
- NVIDIA A100/H100 - Enterprise grade
- NVIDIA V100 - Use `--dtype float32`
- NVIDIA L40S - Enterprise grade

## Model Recommendations

### For Extraction (full pipeline):
1. **Qwen2.5-7B-Instruct** - Best overall accuracy
2. **NuExtract** - Purpose-built for extraction
3. **Llama-3.2-3B-Instruct** - Lightweight option
4. **Mistral-7B-Instruct-v0.3** - Good alternative

### Notes on Model Selection:
- Qwen2.5-VL (OCRFlux) is a vision model and not optimal for pure text extraction
- Using a dedicated text LLM improves extraction accuracy
- The simple script works but may have lower accuracy than using separate models

## Example Output

```json
{
  "invoice_number": "INV-2024-001",
  "invoice_date": "2024-01-15",
  "vendor": {
    "name": "Acme Corporation",
    "address": "123 Business St, City, State 12345"
  },
  "customer": {
    "name": "Customer Inc",
    "address": "456 Client Ave, Town, State 67890"
  },
  "line_items": [
    {
      "description": "Professional Services",
      "quantity": 10,
      "unit_price": 150.00,
      "total": 1500.00
    }
  ],
  "subtotal": 1500.00,
  "tax_amount": 120.00,
  "total_amount": 1620.00,
  "currency": "USD"
}
```

## Memory Optimization Tips

### If you encounter GPU memory errors:
1. **Reduce GPU memory utilization**:
   ```bash
   python extract_invoice_data.py invoice.pdf --gpu-memory 0.6
   ```

2. **Use smaller extraction models**:
   ```bash
   # Instead of 7B model, use 3B
   python extract_invoice_data.py invoice.pdf \
       --extract-model Qwen/Qwen2.5-3B-Instruct
   ```

3. **For V100 or older GPUs**:
   ```bash
   python extract_invoice_data.py invoice.pdf \
       --dtype float32 \
       --gpu-memory 0.7
   ```

## Troubleshooting

1. **GPU Memory Issues**:
   - Reduce `--gpu-memory` parameter (try 0.6 or 0.5)
   - Use smaller models (3B instead of 7B)
   - Close other GPU applications

2. **No GPU Detected**:
   - Ensure NVIDIA drivers are installed
   - Check CUDA installation with `nvidia-smi`

3. **V100 GPU Errors**:
   - Add `--dtype float32` flag (V100 doesn't support bfloat16)

4. **JSON Parsing Errors**:
   - The model may need temperature adjustment
   - Try a different extraction model

5. **Model Download Issues**:
   - Models are auto-downloaded from HuggingFace on first use
   - Ensure internet connection is stable
   - Check disk space (need ~20GB free)