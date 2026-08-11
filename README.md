# ImageToEquation

ImageToEquation converts raster images into purely mathematical Abstract Syntax Trees (ASTs). Instead of storing a grid of pixels (like JPEG/PNG), it mathematically extracts a continuous formula—built from exact polynomial projections, spatial partitions, and trigonometric fields—whose procedural output mathematically resolves into the original image when plotted over an `(x, y)` coordinate plane.

## The Breakthroughs

The project initially attempted to use Neural-Symbolic gradient descent to force images into a highly restrictive mathematical grammar (the Yeganeh Vocabulary). It failed to capture photorealism without cheating via frequency-domain hacks. 

The architecture has since been completely overhauled into a zero-shot, pure-mathematics procedural graphics engine featuring two state-of-the-art mathematical encoding paradigms:

### 1. Algebraic Halftoning
Bypassing traditional frequency-domain (DCT) limitations entirely, this engine treats visual entropy as a continuous algebraic plane. It calculates a massive exact mathematical projection to fit a continuous 2D surface (a Degree-40 polynomial with 800+ terms, stabilized by Ridge Regression) to raster pixel data, and maps it flawlessly onto a single procedural trigonometric curve.

### 2. Procedural QuadTree Vector Graphics
A recursive Binary Space Partitioning (QuadTree) algorithm that structurally parses image data into pure, non-overlapping geometric Boolean intersections (`sdfmask`). This creates a resolution-independent procedural Vector Graphics engine capable of generating colossal 200,000-character, 3,460-node ASTs that evaluate in parallel on the GPU without memory overflow.

## Project Structure

| Directory | Purpose |
|------|---------|
| `assets/` | Original raw images and testing textures. |
| `docs/` | Formatting specs, vocabularies, and massive LaTeX mathematical proofs. |
| `src/image_to_equation/` | Core PyTorch mathematical engine (Evaluator, AST Compiler, Shaders). |
| `experiments/` | Scripts outlining our structural algebraic and geometric breakthroughs. |
| `outputs/` | Rendered PNG images and colossal raw AST text files. |
| `tests/` | Benchmark testing and file extraction harnesses. |

## Running the Mathematical Extractors

The legacy Neural-Symbolic scripts have been removed. You can now execute the true mathematical projections directly from the `experiments/` directory:

**Generate a Procedural QuadTree AST (Vector Graphics):**
```bash
python3 experiments/geometric_quadtree/generate_quadtree_ast.py
```

**Generate a Continuous Algebraic Projection:**
```bash
python3 experiments/algebraic_halftoning/generate_algebraic_polynomial.py
```

The output AST strings and rendered images will be deposited in the `outputs/` directory.

## License

GPL-3.0
