# UNI-D² Research Codebase

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://nkalyanv99.github.io/UNI-D2/)

<p align="center">
  <img src="docs/assets/Logo_Uni_D2.jpeg" width="700">
</p>

Moving beyond the constraints of autoregressive modeling, UNI-D² brings the benefits of iterative refinement—data efficiency, bidirectional context, and parallel decoding—to the text domain. This library serves as the missing foundation for this rapidly growing space, featuring an easily extendible architecture that supports multiple modular training methods. Our unified pipeline facilitates rapid experimentation, enables comparable evaluation, and standardizes benchmarks across the field.

## Highlights
- Hydra + Lightning entry point (`python -m discrete_diffusion`) for experimenting with MDLM, UDLM, BD3LM, FlexMDM, GIDD, SEDD, PartitionMDLM, and CANDI papers.
- Sampling helpers that cover absorbing, BD3LM, GIDD, partition, uniform, autoregressive, FlexMDM, and CANDI samplers plus a reusable `scripts/generate_samples.sh` wrapper.
- Scripts that reproduce training recipes for datasets such as LM1B, OWT, and Text8.

## Getting Started

### Installation
```bash
pip install -e .
```

If you want to isolate dependencies in a Conda env, create/activate it before running the editable install:
```bash
conda create -n uni-d2 python=3.11
conda activate uni-d2
pip install -e .
```

For systems with Flash Attention (CUDA 11.4+), install it after the editable install to boost throughput:
```bash
pip install flash-attn --no-build-isolation
```

For optimized cross-entropy computation on CUDA devices, install Liger Kernel:
```bash
pip install liger-kernel
```

For development you can install the extras that power testing and docs:
```bash
pip install -e "[dev]"
pip install flash-attn --no-build-isolation  # optional
```

The `pyproject.toml`/`requirements.txt` pair declare the dependencies that power training, evaluation, and sampling.

### Configuration

The library caches datasets and artifacts in `~/.cache/discrete_diffusion` by default. You can customize this location in two ways:

1. **Environment Variable (Recommended):**
   ```bash
   export DISCRETE_DIFFUSION_SCRATCH_DIR="/path/to/your/data"
   ```

2. **Command Line Override:**
   ```bash
   python -m discrete_diffusion ... scratch_dir=/path/to/your/data
   ```

## How to run

### Training experiments
Run the Hydra-powered CLI exported at `src/discrete_diffusion/__main__.py` with dataset/model/algorithm overrides. A minimal example:
```bash
PYTHONPATH=src python -u -m discrete_diffusion \
  data=openwebtext-split \
  model=small \
  algo=mdlm \
  loader.batch_size=32 \
  trainer.devices=8 \
  hydra.run.dir=./outputs/owt/mdlm
```
The `examples/` directory contains dataset-specific recipes (e.g., `bd3lm/owt.sh`, `udlm/text8.sh`). Override any Hydra config key by appending `key=value` pairs on the command line.

### Generating samples
Once you have a checkpoint, use the evaluation script:
```bash
PYTHONPATH=src python -m discrete_diffusion.evaluations.generate_samples \
  checkpoint_path=outputs/owt/bd3lm_block16_debug/checkpoints/last.ckpt \
  num_samples=16 \
  num_steps=2000
```


## Repository Structure
- `configs/`: Hydra configuration tree for datasets, models, and learners.
- `examples/`: Scripts and notebooks that reproduce experiments and visualizations.
- `src/discrete_diffusion`: Entry points, Hydra CLI, and the discrete diffusion training API.
- `outputs/`: Default Hydra root for logged checkpoints and metrics.
- `docs/`: Supporting documentation for research artifacts.
- `pyproject.toml` / `requirements.txt`: Dependency and tooling metadata.

## Papers Implemented
1. [MDLM](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf) – Sahoo, Subham, et al., *Simple and effective masked diffusion language models*, NeurIPS 2024.
2. [UDLM](https://arxiv.org/pdf/2412.10193) – Schiff, Yair, et al., *Simple guidance mechanisms for discrete diffusion models*, arXiv 2024.
3. [FlexMDM](https://arxiv.org/pdf/2509.01025) – Kim, Jaeyeon, et al., *Any-Order Flexible Length Masked Diffusion*, arXiv 2025.
4. [Block Diffusion](https://arxiv.org/pdf/2503.09573) – Arriola, Marianne, et al., *Block diffusion: Interpolating between autoregressive and diffusion language models*, arXiv 2025.
5. [GIDD](https://arxiv.org/pdf/2503.04482) – von Rütte, Dimitri, et al., *Generalized interpolating discrete diffusion*, arXiv 2025.
6. [SEDD](https://arxiv.org/pdf/2310.16834) – Lou, Aaron, Chenlin Meng, and Stefano Ermon, *Discrete diffusion modeling by estimating the ratios of the data distribution*, arXiv 2023.
7. [PartitionMDLM](https://arxiv.org/pdf/2505.18883) – Deschenaux, Justin, Lan Tran, and Caglar Gulcehre, *Partition Generative Modeling: Masked Modeling Without Masks*, arXiv 2025.
8. [CANDI](https://arxiv.org/abs/2510.22510) – Pynadath, Patrick, Jiaxin Shi, and Ruqi Zhang, *CANDI: Hybrid Discrete-Continuous Diffusion Models*, arXiv 2025.

## Contributing
Our goal is to maintain this repository as the unified starting point for future research in discrete diffusion for text, keeping it ever-growing and relevant as the field progresses. As such, it will effectively *always* be a work in progress. We welcome any contributions to help it evolve—ranging from full paper implementations with benchmarking to recommendations for features to be added.

## Acknowledgements
We are grateful to the authors of the respective papers for open-sourcing their codebases, which served as a foundation for this library.

## Citation

If you use UNI-D² in your research, please cite it using the following BibTeX entry:

```bibtex
@software{nadimpalli2025unid2,
  author = {Nadimpalli, Kalyan Varma and Pauline, Vincent and Kapl, Ferdinand and Karimi-Mamaghan, Amir Mohammad and Tong, Alexander and Dittadi, Andrea and Bauer, Stefan},
  title = {{UNI-D²: A Unified Codebase for Discrete Diffusion Language Models}},
  version = {0.1.0},
  url = {https://github.com/nkalyanv99/UNI-D2},
  year = {2025}
}
```

## License

```
MIT License

Copyright (c) 2025 Kalyan Varma Nadimpalli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
