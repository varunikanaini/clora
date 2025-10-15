# CLoRA

**CLoRA: Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation**

This repository contains the implementation of CLoRA, a method for composing multiple LoRA (Low-Rank Adaptation) models at test time using contrastive learning for improved image generation with Stable Diffusion.

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver that ensures consistent dependency resolution.

#### 1. Install UV

If you don't have UV installed:

```bash
# Install UV using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv

# Or using pipx
pipx install uv
```

#### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/gemlab-vt/clora
cd CLoRA

# Create virtual environment and install dependencies
uv sync
```

#### 3. Verify Installation

```bash
# Check if everything is installed correctly
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Running the Jupyter Notebook

The main demonstration and examples are provided in the Jupyter notebook.

```bash
# Start Jupyter Notebook using UV
uv run jupyter notebook notebook.ipynb

# Or start Jupyter Lab
uv run jupyter lab notebook.ipynb
```

## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{Meral_2025_ICCV,
    author    = {Meral, Tuna Han Salih and Simsar, Enis and Tombari, Federico and Yanardag, Pinar},
    title     = {Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {18090-18100}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

- **Author**: Tuna Han Salih Meral
- **Email**: tmeral@vt.edu

For questions about the implementation or research, please open an issue or contact the author directly.
