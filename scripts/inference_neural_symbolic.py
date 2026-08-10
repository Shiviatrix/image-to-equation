"""Run a trained ImageToAST checkpoint and render its best safe equation.

Example:
    python inference_neural_symbolic.py test_images/natural_einstein.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from image_to_equation.evaluator import GridEvaluator
from image_to_equation.multiscale_fitter import psnr, ssim
from image_to_equation.neural_symbolic import ImageToAST, YeganehVocabulary, program_to_source


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[ImageToAST, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "image_to_ast_v1":
        raise ValueError(f"unsupported checkpoint format in {checkpoint_path}")
    vocabulary = YeganehVocabulary(constant_bins=int(checkpoint["vocabulary_bins"]))
    model = ImageToAST(vocabulary=vocabulary).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint.get("image_size", 64))


@torch.no_grad()
def select_rate_distortion_candidate(
    model: ImageToAST,
    target: np.ndarray,
    beam_size: int,
    source_budget: int | None,
    length_penalty: float,
) -> tuple[list[str], str, np.ndarray, dict[str, float | int]]:
    """Choose the best rendered beam under a hard executable-source budget.

    The input image is the reconstruction target, so evaluating a small beam
    against it is legitimate test-time optimisation.  Passing ``None`` for
    ``source_budget`` makes this a pure visual-quality selector.
    """
    device = next(model.parameters()).device
    model_input = torch.from_numpy(target).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    candidates = model.generate(model_input, beam_size=beam_size)
    evaluator = GridEvaluator()
    target_tensor = torch.from_numpy(target).float().div(255.0)
    ranked: list[tuple[float, list[str], str, np.ndarray, float, float]] = []
    for program in candidates:
        source = program_to_source(program, model.vocabulary)
        if source_budget is not None and len(source) > source_budget:
            continue
        decoded = evaluator.evaluate(source, target.shape[1], target.shape[0], render_mode="linear").cpu().numpy()
        decoded_tensor = torch.from_numpy(decoded).float().div(255.0)
        psnr_value, ssim_value = psnr(decoded_tensor, target_tensor), ssim(decoded_tensor, target_tensor)
        score = psnr_value - length_penalty * len(source)
        ranked.append((score, program, source, decoded, psnr_value, ssim_value))
    if not ranked:
        raise RuntimeError("no valid beam candidate was produced")
    _, program, source, decoded, psnr_value, ssim_value = max(ranked, key=lambda item: item[0])
    return program, source, decoded, {
        "psnr": psnr_value, "ssim": ssim_value, "candidate_count": len(ranked),
        "source_budget": source_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/image_to_ast.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("neural_inference_output"))
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--source-budget", type=int, default=None, help="optional maximum equation characters")
    parser.add_argument("--length-penalty", type=float, default=0.0, help="PSNR penalty per source character")
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = _load_model(args.checkpoint, device)
    original_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise SystemExit(f"unable to read raster: {args.image}")
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    target = cv2.resize(original, (image_size, image_size), interpolation=cv2.INTER_AREA)
    program, source, decoded, metrics = select_rate_distortion_candidate(
        model, target, args.beam_size, args.source_budget, args.length_penalty
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    decoded_path = args.output_dir / f"decoded_{stem}.png"
    comparison_path = args.output_dir / f"comparison_{stem}.png"
    source_path = args.output_dir / f"equation_{stem}.txt"
    metadata_path = args.output_dir / f"result_{stem}.json"
    cv2.imwrite(str(decoded_path), cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(comparison_path), cv2.cvtColor(np.hstack((target, decoded)), cv2.COLOR_RGB2BGR))
    source_path.write_text(source + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps({
        "image": str(args.image), "checkpoint": str(args.checkpoint), "image_size": image_size,
        "tokens": program, "token_count": len(program), "source_length": len(source), **metrics,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"tokens={len(program)} source_chars={len(source)} budget={args.source_budget} candidates={metrics['candidate_count']} PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
    print(f"comparison={comparison_path}")
    print(f"equation={source_path}")


if __name__ == "__main__":
    main()
