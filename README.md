# ImageToEquation

ImageToEquation converts normal images into mathematical formulas. Instead of storing a grid of pixels like a JPEG or PNG does, it creates a long math equation using basic algebra and trigonometry. When you graph this equation, it draws the original image.

## How It Works

At first, this project tried to use machine learning to guess the math formulas, but it struggled with complex photos. We have since completely rewritten the project to use two exact math methods instead of AI:

### 1. The Polynomial Method
This method treats the image like a 3D landscape and calculates a massive curve (a 40th-degree polynomial) that perfectly matches the brightness of the image. It then uses a simple cosine wave to draw the final result. It creates smooth, continuous math art.

### 2. The Box-Splitting Method (QuadTree)
This method looks at the image and splits it into smaller and smaller boxes based on how much detail is in each area. It then combines all these boxes into one giant math equation. This method can handle highly detailed photos and creates formulas that are hundreds of thousands of characters long.

## Project Structure

| Directory | Purpose |
|------|---------|
| `assets/` | Original images used for testing. |
| `docs/` | Project rules and formats. |
| `src/image_to_equation/` | The core code that turns math into images. |
| `experiments/` | The scripts that generate the math equations. |
| `outputs/` | The generated images and text files containing the long equations. |
| `tests/` | Code for testing the project. |

## Running the Code

You can generate the math equations by running the scripts in the `experiments/` directory:

**Generate an equation using the Box-Splitting Method:**
```bash
python3 experiments/geometric_quadtree/generate_quadtree_ast.py
```

**Generate an equation using the Polynomial Method:**
```bash
python3 experiments/algebraic_halftoning/generate_algebraic_polynomial.py
```

The output text files and the final drawn images will be saved in the `outputs/` directory.

## License

GPL-3.0
