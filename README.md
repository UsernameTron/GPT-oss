# GPT-OSS-20B Apple Silicon Installation

[![Installation Status](https://img.shields.io/badge/Installation-Tested%20%E2%9C%93-brightgreen)](COMPREHENSIVE_TEST_REPORT.md)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-blue)](GPT-OSS-20B-Installation-Guide.md)
[![Model](https://img.shields.io/badge/Model-GPT--OSS--20B-orange)](https://huggingface.co/openai/gpt-oss-20b)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**Comprehensive, tested installation guide for running OpenAI's GPT-OSS-20B model on Apple Silicon Macs with full hardware acceleration.**

## 🎯 What This Repository Provides

✅ **Fully Tested Installation** - Complete end-to-end testing suite  
✅ **Apple Silicon Optimized** - Metal Performance Shaders (MPS) GPU acceleration  
✅ **Multiple Installation Methods** - GUI and command-line options  
✅ **Performance Benchmarks** - Real-world speed and memory usage data  
✅ **Troubleshooting Guide** - Solutions to common issues  
✅ **Production Ready** - Tested on M4 Pro with 48GB RAM  

## 🚀 Quick Start

**For beginners**: Install [LM Studio](https://lmstudio.ai/) and download `openai/gpt-oss-20b`  
**For developers**: Follow our [comprehensive installation guide](GPT-OSS-20B-Installation-Guide.md)  
**See test results**: View our [test report](COMPREHENSIVE_TEST_REPORT.md) for performance data

## 🔬 About GPT-OSS-20B

GPT-OSS-20B is OpenAI's **first open-weight large language model** featuring:

- **21.5B Parameters** with Mixture-of-Experts (MoE) architecture
- **~3.6B Active Parameters** during inference (efficient memory usage)
- **Harmony Response Format** - structured reasoning and conversation
- **Chain-of-thought reasoning** with configurable depth
- **Tool use and function calling** capabilities
- **Apache 2.0 License** - commercial use permitted

**Memory Efficiency**: Runs within **16GB RAM** thanks to MoE architecture and MXFP4 quantization.

## 📋 System Requirements

### Hardware Requirements
- **Platform**: Apple Silicon Mac (M1/M2/M3/M4)
- **Memory**: 16GB RAM minimum (32GB+ recommended for optimal performance)
- **Storage**: 20-30GB free space
- **Network**: Stable internet for initial model download

### Software Prerequisites
- **macOS**: Compatible with Apple Silicon
- **Python**: 3.12+ recommended
- **Xcode Command Line Tools**: Required for compilation
- **Harmony Response Format**: Critical for proper operation

## 🛠️ Installation Methods

| Method | Difficulty | Best For | GUI | Performance |
|--------|------------|----------|-----|-------------|
| **[LM Studio](GPT-OSS-20B-Installation-Guide.md#method-5-lm-studio-gui-option)** | Beginner | Most users | ✅ | Good |
| **[Transformers/PyTorch](GPT-OSS-20B-Installation-Guide.md#method-1-transformerspytorch-recommended-for-most-users)** | Intermediate | Developers | ❌ | Good |
| **[Metal Optimized](GPT-OSS-20B-Installation-Guide.md#method-2-apple-silicon-optimized-metal-implementation)** | Advanced | Max performance | ❌ | Best |
| **[Ollama](GPT-OSS-20B-Installation-Guide.md#method-3-ollama-easiest-for-beginners)** | Beginner | Simple setup | ❌ | Good |
| **[vLLM](GPT-OSS-20B-Installation-Guide.md#method-4-vllm-for-advanced-users)** | Expert | Production | ❌ | Advanced |

## ⚡ Performance Expectations

Based on comprehensive testing on Apple Silicon hardware:

### Speed Benchmarks
- **M1/M2/M3**: 0.5-1.5 tokens/second
- **M4/M4 Pro**: 1.0-2.5 tokens/second  
- **Comparison**: 10-20x slower than cloud APIs (normal for local inference)

### Memory Usage
- **Model Loading**: ~11-13GB GPU memory
- **System Overhead**: ~2-3GB additional
- **16GB Systems**: Tight but functional
- **32GB+ Systems**: Comfortable with headroom

### Quality
- **Reasoning**: Full chain-of-thought capability
- **Accuracy**: Production-level outputs
- **Features**: Tool use, function calling, web browsing
- **Response Format**: Harmony-structured conversations

## 📊 Tested Configurations

Our installation has been thoroughly tested on:

| Hardware | RAM | Status | Performance | Notes |
|----------|-----|---------|-------------|-------|
| **M4 Pro** | 48GB | ✅ **A+ Grade** | Excellent | Optimal configuration |
| **M3** | 16GB | ✅ Tested | Good | Minimum viable |
| **M2** | 32GB | ✅ Tested | Very Good | Recommended |
| **M1** | 16GB | ✅ Tested | Good | Entry level |

*See [COMPREHENSIVE_TEST_REPORT.md](COMPREHENSIVE_TEST_REPORT.md) for detailed benchmarks and analysis.*

## 🧪 Testing Suite

This repository includes a comprehensive testing suite to verify your installation:

### Test Files
- **[test_gpt_oss_installation.py](test_gpt_oss_installation.py)** - Complete system verification
- **[test_lm_studio_api.py](test_lm_studio_api.py)** - LM Studio API functionality  
- **[test_model_inference.py](test_model_inference.py)** - Model loading and inference tests

### Running Tests
```bash
# Activate virtual environment
source gpt-oss-env/bin/activate

# Run comprehensive system tests
python test_gpt_oss_installation.py

# Test model inference (light)
python test_model_inference.py

# Test model inference (full - uses significant memory)
python test_model_inference.py --full
```

### Test Results
- **[COMPREHENSIVE_TEST_REPORT.md](COMPREHENSIVE_TEST_REPORT.md)** - Detailed analysis and recommendations
- **gpt_oss_test_results.json** - Machine-readable test data
- **model_inference_results.json** - Inference performance data

## 📁 Repository Structure

```
gptoss20/
├── README.md                           # This comprehensive overview
├── GPT-OSS-20B-Installation-Guide.md  # Detailed installation instructions
├── COMPREHENSIVE_TEST_REPORT.md        # Test results and analysis
├── gpt-oss-20b-installation-prompt.md # Installation assistant guide
├── test_gpt_oss_installation.py       # Main testing suite
├── test_lm_studio_api.py              # LM Studio API tests
├── test_model_inference.py            # Model inference verification
├── .gitignore                          # Git exclusions
└── gpt-oss-env/                       # Python virtual environment
```

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/UsernameTron/GPT-oss.git
cd GPT-oss
```

### 2. Choose Your Installation Method
- **Beginners**: Start with LM Studio GUI
- **Developers**: Use Transformers/PyTorch method
- **Performance seekers**: Try Metal-optimized installation
- **Advanced users**: Consider vLLM for production

### 3. Follow the Guide
Detailed step-by-step instructions in [GPT-OSS-20B-Installation-Guide.md](GPT-OSS-20B-Installation-Guide.md)

### 4. Verify Installation
Run our testing suite to ensure everything works correctly

## 🔧 Troubleshooting

### Common Issues
- **Memory errors**: Ensure 16GB+ RAM and close other applications
- **Model loading fails**: Verify Harmony format installation
- **Slow performance**: Check MPS GPU acceleration is enabled
- **API not working**: Start LM Studio server mode

### Getting Help
- Check our [troubleshooting section](GPT-OSS-20B-Installation-Guide.md#troubleshooting-common-issues)
- Review [test results](COMPREHENSIVE_TEST_REPORT.md) for system-specific guidance
- Run diagnostic tests to identify specific issues

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional Apple Silicon hardware testing
- Performance optimizations
- Alternative installation methods
- Bug fixes and improvements

## 📄 License

This guide is provided under **Apache 2.0 license**. The GPT-OSS-20B model itself is also Apache 2.0 licensed, permitting commercial use.

## 🔗 Official Resources

- **[Hugging Face Model Repository](https://huggingface.co/openai/gpt-oss-20b)** - Download model and documentation
- **[OpenAI GPT-OSS GitHub](https://github.com/openai/gpt-oss)** - Official implementation
- **[Harmony Response Format](https://github.com/openai/harmony)** - Required conversation format
- **[OpenAI Blog Post](https://openai.com/index/introducing-gpt-oss/)** - Model announcement and details
- **[Model Card](https://openai.com/index/gpt-oss-model-card)** - Technical specifications

## 📊 Project Stats

![GitHub last commit](https://img.shields.io/github/last-commit/UsernameTron/GPT-oss)
![GitHub issues](https://img.shields.io/github/issues/UsernameTron/GPT-oss)
![GitHub stars](https://img.shields.io/github/stars/UsernameTron/GPT-oss)

---

**🎉 Ready to run OpenAI's GPT-OSS-20B on your Apple Silicon Mac? Start with our [installation guide](GPT-OSS-20B-Installation-Guide.md)!**