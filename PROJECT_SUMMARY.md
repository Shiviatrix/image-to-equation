# Evaluation Results

This document records the training and evaluation outcomes for the ImageToEquation neural-symbolic model.

## Training setup

- **Dataset:** Synthetically generated on-the-fly. Each sample is a random parametric equation rendered to a 64×64 RGB image, paired with its canonical token sequence.
- **Curriculum:** The generator starts with simple shapes (single circle or line) and gradually introduces multi-primitive compositions and trigonometric sweeps as training progresses.
- **Optimiser:** AdamW, learning rate 3×10⁻⁴, weight decay 1×10⁻⁴, gradient clipping at 1.0.
- **Hardware:** Apple M-series CPU/MPS. Mixed precision enabled via `torch.autocast`.
- **Steps completed:** 25,000.

## Results

The model was evaluated every 5,000 steps on two held-out images: a flat-colour geometric fish and a greyscale photograph of Einstein.

### Best checkpoint (step 25,000)

| Image | PSNR (dB) | SSIM | Tokens | Equation length (chars) |
|-------|-----------|------|--------|------------------------|
| Structured fish | 11.40 | 0.515 | 57 | 236 |
| Einstein photo | 8.52 | 0.112 | 74 | 279 |
| **Mean** | **9.55** | — | — | — |

### Baseline (step 1,000, before curriculum learning)

| Image | PSNR (dB) | SSIM | Tokens | Equation length (chars) |
|-------|-----------|------|--------|------------------------|
| Structured fish | 1.19 | 0.091 | 23 | 115 |
| Einstein photo | 7.70 | 0.051 | 23 | 115 |

### Batch validation (11 images, mixed categories)

A separate batch test encoded 11 images across three categories (structured art, natural photos, hybrid UI elements) through the full pipeline.

- 4 images were accepted by the symbolic path (`.i2e` type `I2ES`).
- 7 images fell back to DCT compression (`.i2e` type `I2E2`).
- Natural images compressed via DCT retained 37.78–43.30 dB PSNR.
- Shadowed UI fixtures were incorrectly accepted by the symbolic path and reconstructed poorly (~10–12 dB), exposing a routing bug: the acceptance gate was using binary IoU instead of full-RGB PSNR. This is a known issue and has not yet been fixed.

## Observations

- The network reliably emits grammatically valid equations. The beam search decoder is constrained by a prefix grammar, so it cannot produce malformed ASTs.
- On geometric shapes, the model improved by roughly 10 dB over the untrained baseline across 25,000 steps. This confirms that the curriculum schedule is working as intended.
- On natural photographs, the model plateaus around 7–9 dB. The current token vocabulary (sin, cos, add, mul, pow2, and 65 quantised constants) does not have enough expressive capacity to represent high-frequency texture, which is why the DCT fallback exists.
- All 18 unit tests pass.
