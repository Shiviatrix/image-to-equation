# Architectural Evaluation & Results

This document records the evolution and evaluation outcomes for the ImageToEquation project, detailing the pivot from a constrained Neural-Symbolic architecture to a robust, Zero-Shot Procedural mathematical engine.

## Phase 1: Neural-Symbolic (Deprecated)

Initially, the project attempted to train a CNN-Transformer model to emit Yeganeh Vocabulary tokens via Gradient Descent.

### Results
- The model successfully learned primitive geometric compositions.
- **Bottleneck:** It flatlined at 7-9 dB PSNR on high-frequency natural photographs (e.g., Einstein).
- **Fundamental Flaw:** Gradient descent was too inefficient for combinatorial symbolic grammar. To compensate, a frequency-domain Discrete Cosine Transform (DCT) fallback was implemented. However, falling back to a grid of cosine waves defeated the philosophical goal of encoding images as pure continuous mathematical equations.
- *Status: Deprecated. The training scripts have been removed.*

---

## Phase 2: Zero-Shot Mathematical Generation (Active)

To shatter the algorithmic bottleneck, we mathematically proved that you can project raster pixels into the AST without gradient descent. We developed two successful paradigms:

### 1. Algebraic Halftoning
- **Method:** Solves an exact analytical Least Squares projection to map visual entropy onto a continuous 2D plane using a massive high-degree polynomial.
- **Implementation:** Fits a Degree-40 polynomial (861 terms) stabilized via Ridge Regression, evaluated procedurally into a single trigonometric stroke `cos(200 * sqrt(x^2 + y^2))`.
- **Result:** Beautiful continuous mathematical rendering with zero reliance on frequency-domain hacks.

### 2. Geometric QuadTree Vector Partitioning
- **Method:** Recursively parses an image into Binary Space Partitions based on pixel variance.
- **Implementation:** Formulates a massive Boolean geometric AST using exact non-overlapping intersections (`sdfmask(box(...))`).
- **Result:** Successfully compiles and evaluates a 200,000-character, 3,460-node spatial AST via PyTorch without memory overflow, yielding near-photoreal vector reconstruction.

## System Health
- **Evaluator Limits:** The `GridEvaluator` compiler handles extremely deep recursive ASTs up to 200,000 characters by overriding Python C stack limits.
- **Mathematical Integrity:** All equations conform strictly to the Yeganeh AST ruleset without resorting to explicit raster storage.
