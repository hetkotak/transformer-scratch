# transformer-scratch
Creating transformers based LLM from scratch

## 🎯 Project Overview

This repository contains a comprehensive implementation of transformer-based language models built from scratch using PyTorch. The goal is to provide a clear, educational journey through the core components of modern LLMs while maintaining clean, well-documented code that others can learn from.

## 📁 Project Structure

```
transformer-scratch/
├── src/transformer/              # Core transformer components
│   ├── scaled_dot_product_attention.py  # Core attention mechanism  
│   ├── multihead_attention.py    # Multi-head attention wrapper
│   ├── layers.py                 # Transformer layers & blocks
│   ├── embeddings.py             # Position & token embeddings
│   ├── models.py                 # Complete model architectures
│   └── utils.py                  # Utility functions
├── examples/                     # Usage examples & tutorials
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks for exploration
└── PROJECT_PLAN.md              # Detailed roadmap
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/hetkotak/transformer-scratch.git
   cd transformer-scratch
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run examples**
   ```bash
   python examples/01_scaled_dot_product_attention.py
   python examples/02_multihead_attention.py
   ```

## 📚 Current Progress

- [x] Project structure setup
- [x] Scaled dot-product attention implementation
- [ ] Multi-head attention implementation
- [ ] Transformer layers
- [ ] Complete GPT model

## 🛠️ Development

Run tests:
```bash
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
