# ImageToEquation

ImageToEquation converts images into mathematical equations. Instead of storing a grid of pixels, it finds a formula — built from sines, cosines, and basic algebra — whose output looks like the original image when plotted over an (x, y) coordinate plane.

The system has two paths:

- **Symbolic path:** A small neural network looks at an image and predicts a short trigonometric equation that approximates it. The equation vocabulary is modelled after the parametric art of Hamid Naderi Yeganeh.
- **DCT path:** When the symbolic path can't hit the quality bar (common for photographs with lots of texture), the image is compressed using a standard block-based Discrete Cosine Transform instead.

Both paths output a custom `.i2e` binary file that can be decoded back into an image.

## How it works

1. A CNN encoder extracts spatial features from the input image.
2. A Transformer decoder reads those features and emits a sequence of tokens representing an Abstract Syntax Tree (AST) of mathematical operations.
3. A differentiable evaluator compiles that AST into PyTorch tensor operations so we can render the equation and compute pixel-level loss against the target.
4. If the rendered equation meets the PSNR/SSIM quality threshold, we keep it. Otherwise the image is routed to the DCT fallback.

## Stack

- Python 3.13
- PyTorch (CNN, Transformer, mixed-precision training via `torch.autocast`)
- SciPy (DCT frequency transforms)
- pytest (test suite)

## Project structure

| File | Purpose |
|------|---------|
| `neural_symbolic.py` | CNN encoder, Transformer decoder, grammar-constrained beam search |
| `evaluator.py` | AST-to-PyTorch compiler with SDF rendering and caching |
| `symbolic_fitter.py` | Greedy MDL shape fitter (superellipses, capsules, wedges) |
| `multiscale_fitter.py` | Laplacian-pyramid residual fitting with smooth blending |
| `procedural_shaders.py` | Parametric colour fields (gradients, lattice noise) |
| `dct_extractor.py` | Block DCT extraction and quantisation |
| `i2e_encoder.py` / `i2e_decoder.py` | Binary `.i2e` file format read/write |
| `synthetic_yeganeh_dataset.py` | On-the-fly training data generator |
| `train_neural_symbolic.py` | Training loop with curriculum scheduling and validation |
| `inference_neural_symbolic.py` | CLI for running a trained model on new images |
| `batch_test.py` | Batch encode-decode benchmark across image categories |
| `run_all.py` / `main.py` | Top-level pipeline that routes between symbolic and DCT |

## Running

**Train the neural-symbolic model:**
```bash
python3 train_neural_symbolic.py --steps 25000
```

**Run inference on an image:**
```bash
python3 inference_neural_symbolic.py --image path/to/image.png --checkpoint checkpoints/best_yeganeh_model.pt
```

**Run the test suite:**
```bash
pytest tests/ -v
```

## License

GPL-3.0
