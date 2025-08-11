# GPT-OSS-20B Installation Guide for Apple Silicon

## Overview
GPT-OSS-20B is OpenAI's open-weight 21.5B parameter model with configurable reasoning capabilities using a Mixture-of-Experts (MoE) architecture. Only ~3.6B parameters are active during inference, combined with MXFP4 8-bit quantization, allowing the model to run within 16GB of memory.

**Important Performance Expectations**: This model is designed for experimentation and learning on Apple Silicon, not production use. Expect generation speeds of 0.5-2 tokens per second on M1/M2/M3 hardware - significantly slower than dedicated GPU systems.

**Model Repository**: https://huggingface.co/openai/gpt-oss-20b

## System Requirements

### Hardware Requirements
- **Memory**: Minimum 16GB RAM (model runs within 16GB)
- **Storage**: ~20-30GB free space (depending on installation method)
- **Platform**: Apple Silicon Mac (M1/M2/M3/M4)

### Software Prerequisites
- **macOS**: Compatible with Apple Silicon
- **Python**: Version 3.12 recommended
- **Xcode Command Line Tools**: Required for compilation
- **Git**: For cloning repositories
- **Git LFS**: For large file handling (if using git clone method)

## Pre-Installation Setup

### 1. Install Xcode Command Line Tools
```bash
xcode-select --install
```

### 2. Verify Python Installation
Ensure Python 3.12 is installed. If not, install via:
- Homebrew: `brew install python@3.12`
- Official Python installer from python.org
- pyenv for version management

### 3. Install Git LFS (if needed)
```bash
brew install git-lfs
git lfs install
```

### 4. Set Up Python Virtual Environment (Recommended)
```bash
python3.12 -m venv gpt-oss-env
source gpt-oss-env/bin/activate
pip install --upgrade pip
```

## Installation Methods

### Method 1: Transformers/PyTorch (Recommended for Most Users)

#### Step 1: Install Dependencies
```bash
# Install PyTorch with MPS support for Apple Silicon
pip install torch torchvision

# Install Transformers and kernels for GPT-OSS support
pip install -U transformers kernels

# Install Harmony format support (CRITICAL - model will not work without this)
pip install harmony-openai
```

#### Step 2: Download Model Weights
**Option A: Using Hugging Face CLI**
```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub[cli]

# Download the complete model
huggingface-cli download openai/gpt-oss-20b --local-dir gpt-oss-20b/
```

**Option B: Download specific components**
```bash
# Download original weights only
huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

#### Step 3: Install GPT-OSS Package
```bash
pip install gpt-oss
```

#### Step 4: Test Installation
```bash
python -m gpt_oss.chat gpt-oss-20b/
```

### Method 2: Apple Silicon Optimized (Metal Implementation)

#### Step 1: Clone Repository
```bash
git clone https://github.com/openai/gpt-oss.git
cd gpt-oss
```

#### Step 2: Install with Metal Support
```bash
# Critical: Set build flag for proper Metal compilation on Apple Silicon
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"

# Install Harmony format support (CRITICAL)
pip install harmony-openai
```

#### Step 3: Download Metal-Optimized Weights
```bash
hf download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/metal/
```

#### Step 4: Test Metal Implementation
```bash
# Test basic generation
python gpt_oss/metal/examples/generate.py gpt-oss-20b/metal/model.bin -p "why did the chicken cross the road?"

# Or start interactive chat with Metal backend
python -m gpt_oss.chat gpt-oss-20b/metal/model.bin --backend metal
```

### Method 3: Ollama (Easiest for Beginners)

#### Step 1: Install Ollama
Download from: https://ollama.ai/download
Or via Homebrew:
```bash
brew install ollama
```

#### Step 2: Pull GPT-OSS Model
```bash
ollama pull gpt-oss:20b
```

#### Step 3: Start Using
```bash
ollama run gpt-oss:20b
```

### Method 4: vLLM (For Advanced Users)

#### Step 1: Install vLLM with GPT-OSS Support
```bash
uv pip install --pre vllm==0.10.1+gptoss
```

#### Step 2: Serve the Model
```bash
vllm serve openai/gpt-oss-20b
```

### Method 5: LM Studio (GUI Option)

#### Step 1: Download LM Studio
Visit: https://lmstudio.ai/
Download and install the macOS version

#### Step 2: Download Model
```bash
lms get openai/gpt-oss-20b
```

Or use the GUI to search for and download "openai/gpt-oss-20b"

## Critical Configuration Requirements

### Harmony Response Format
**IMPORTANT**: GPT-OSS models require the Harmony response format to function correctly.

#### Install Harmony
```bash
pip install harmony-openai
```

#### Resources for Harmony Format
- **Documentation**: https://github.com/openai/harmony
- **Purpose**: Defines conversation structures and enables reasoning output
- **Usage**: Required for proper model operation - model "will not work correctly" without it

### Configuration Files Location
When downloaded, the model includes several important files:
- `config.json`: Model configuration
- `generation_config.json`: Generation parameters
- `chat_template.jinja`: Chat formatting template
- `tokenizer.json` and related tokenizer files

## File Structure After Installation

```
gpt-oss-20b/
├── README.md
├── LICENSE
├── USAGE_POLICY
├── config.json
├── generation_config.json
├── chat_template.jinja
├── tokenizer files/
├── original/           # Original model weights
│   └── model-*.safetensors
└── metal/              # Apple Silicon optimized
    └── model.bin       # 13.8GB Metal-optimized weights
```

## Verification and Testing

### Basic Functionality Test
Create a simple Python script to test the installation:

```python
# Save as test_gpt_oss.py
from transformers import pipeline
import torch

model_id = "openai/gpt-oss-20b"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```

### Performance Considerations
- **Memory Usage**: Monitor RAM usage during inference (~11-13GB GPU memory)
- **Speed Expectations**: 0.5-2 tokens/second on Apple Silicon (be patient)
- **Temperature Settings**: Adjust for different reasoning levels (Low/Medium/High)
- **Token Limits**: Be mindful of context length limits
- **First Run**: Model loading can be slow initially as weights move to GPU

## Troubleshooting Common Issues

### Memory Issues
- Ensure you have sufficient RAM (16GB minimum)
- Close other memory-intensive applications
- **Memory optimization**: Set environment variable `export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for better MPS memory utilization
- Monitor memory usage with Activity Monitor
- If out of memory: model will automatically offload layers to CPU with `device_map="auto"`

### Installation Failures
- Verify Xcode Command Line Tools installation: `xcode-select --print-path`
- Check Python version compatibility: `python --version` (must be 3.12+)
- Ensure sufficient disk space (~30GB free recommended)
- **Metal compilation issues**: Verify `GPTOSS_BUILD_METAL=1` flag was used

### Model Loading Issues
- Verify all model files downloaded correctly
- Check file permissions
- **Critical**: Ensure Harmony format is properly configured - model will not work without it
- **Performance issues**: Verify MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`
- **CPU fallback**: If model runs on CPU only, check GPU memory availability

### Harmony Format Issues
- Install required package: `pip install harmony-openai`
- Use proper message format with role-based conversations
- Refer to https://github.com/openai/harmony for documentation
- Model outputs may be incorrect without proper Harmony formatting

## Additional Resources

### Official Documentation
- **Model Repository**: https://huggingface.co/openai/gpt-oss-20b
- **GitHub Repository**: https://github.com/openai/gpt-oss
- **Harmony Format**: https://github.com/openai/harmony
- **OpenAI Blog**: https://openai.com/index/introducing-gpt-oss/
- **Model Card**: https://openai.com/index/gpt-oss-model-card

### Model Specifications
- **Parameters**: 21.5B
- **Tensor Types**: BF16, U8
- **License**: Apache 2.0
- **Reasoning Levels**: Configurable (Low, Medium, High)
- **Features**: Chain-of-thought reasoning, tool use, function calling

### Community Resources
- Hugging Face model page discussions
- GitHub issues and discussions
- Community forums and Discord channels

## Usage Notes

### Model Capabilities
- Full chain-of-thought reasoning
- Tool use and function calling
- Web browsing capabilities
- Fine-tuning support
- Native MXFP4 quantization

### Best Practices
- **Always use the Harmony response format** - model will not function correctly without it
- Monitor system resources during inference (use Activity Monitor)
- Keep model files in a dedicated directory with sufficient storage
- Regular updates via `pip install -U` for dependencies
- **Realistic expectations**: This is for experimentation, not production use on Mac hardware
- **GPU preference**: Ensure model uses MPS (Apple GPU) rather than CPU for better performance
- **Memory management**: Close unnecessary applications before running inference

### License and Usage
- Apache 2.0 License
- Refer to USAGE_POLICY file in the model repository
- Commercial use permitted under Apache 2.0 terms

## Advanced Optimization Options (Optional)

### Future Considerations
- **MLX Framework**: Apple's MLX library may offer better performance for large language models on Apple Silicon in the future. Community projects like `mlx-lm` provide 4-bit quantized versions, but these come with accuracy trade-offs and are not recommended for initial installations.
- **Core ML Conversion**: Apple Neural Engine (ANE) utilization through Core ML conversion may become available but is not currently supported out-of-the-box.

### Memory Management Environment Variables
For advanced users experiencing memory issues:
```bash
# Allow MPS to use all available memory
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Disable MPS fallback if troubleshooting
export PYTORCH_ENABLE_MPS_FALLBACK=0
```

## Summary

This installation guide provides multiple pathways to get GPT-OSS-20B running on Apple Silicon systems, from beginner-friendly options like Ollama to advanced setups with Metal optimization. Remember that the **Harmony response format is absolutely critical** for proper operation, and performance expectations should be set appropriately for Mac hardware - this is an experimental and educational tool, not a production-ready solution.