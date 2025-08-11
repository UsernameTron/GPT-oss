# GPT-OSS-20B Apple Silicon Installation

This repository contains comprehensive installation instructions for running OpenAI's GPT-OSS-20B model on Apple Silicon (M1/M2/M3/M4) Mac systems.

## About GPT-OSS-20B

GPT-OSS-20B is OpenAI's open-weight 21.5B parameter model with configurable reasoning capabilities. It uses a Mixture-of-Experts (MoE) architecture with only ~3.6B active parameters during inference, allowing it to run within 16GB of memory on Apple Silicon.

## Quick Start

See the [comprehensive installation guide](GPT-OSS-20B-Installation-Guide.md) for detailed instructions.

### Key Requirements
- Apple Silicon Mac (M1/M2/M3/M4)
- 16GB RAM minimum
- Python 3.12+
- Harmony response format (critical for proper operation)

## Installation Methods

1. **Transformers/PyTorch** - Recommended for most users
2. **Apple Silicon Optimized (Metal)** - Best performance on Mac
3. **Ollama** - Easiest for beginners
4. **vLLM** - Advanced users
5. **LM Studio** - GUI option

## Performance Expectations

This model is designed for experimentation and learning on Apple Silicon, not production use. Expect generation speeds of 0.5-2 tokens per second - significantly slower than dedicated GPU systems.

## License

This guide is provided under Apache 2.0 license. The GPT-OSS-20B model itself is also Apache 2.0 licensed.

## Resources

- [Hugging Face Model Repository](https://huggingface.co/openai/gpt-oss-20b)
- [OpenAI GPT-OSS GitHub](https://github.com/openai/gpt-oss)
- [Harmony Response Format](https://github.com/openai/harmony)