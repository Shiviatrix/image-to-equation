"""Resumable, curriculum-driven continuous training for ImageToAST.

By default this is intentionally an infinite loop.  It writes a latest
checkpoint at every validation interval and a separate best checkpoint only
when the fixed visual validation score improves.  Stop it with Ctrl-C; the
last completed evaluation checkpoint remains recoverable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterable

import cv2
import numpy as np
import torch

from inference_neural_symbolic import select_rate_distortion_candidate
from image_to_equation.neural_symbolic import ImageToAST, YeganehVocabulary, program_to_source, sequence_cross_entropy
from image_to_equation.synthetic_yeganeh_dataset import SyntheticYeganehDataset, collate_programs, curriculum_phase


DEFAULT_VALIDATION_IMAGES = (Path("test_images/structured_control_fish.png"), Path("test_images/natural_einstein.jpg"))


def _checkpoint_payload(
    model: ImageToAST,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    vocabulary: YeganehVocabulary,
    image_size: int,
    seed: int,
    step: int,
    best_score: float,
) -> dict[str, object]:
    return {
        "format": "image_to_ast_v1",
        "vocabulary_bins": vocabulary.constant_bins,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "image_size": image_size,
        "seed": seed,
        "step": step,
        "best_validation_psnr": best_score,
    }


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate_fixed_images(model: ImageToAST, image_size: int, paths: Iterable[Path], source_budget: int | None) -> list[dict[str, object]]:
    """Evaluate generated (not teacher-forced) equations on fixed real images."""
    model.eval()
    metrics: list[dict[str, object]] = []
    for path in paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            metrics.append({"image": str(path), "error": "missing or unreadable"})
            continue
        target = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (image_size, image_size), interpolation=cv2.INTER_AREA)
        try:
            programs, source, _, selection = select_rate_distortion_candidate(
                model, target, beam_size=4, source_budget=source_budget, length_penalty=0.0
            )
            metrics.append({
                "image": str(path), "psnr": selection["psnr"],
                "ssim": selection["ssim"], "tokens": len(programs),
                "source_length": len(source),
            })
        except Exception as error:  # preserve training after a bad candidate
            metrics.append({"image": str(path), "error": f"{type(error).__name__}: {error}"})
    model.train()
    return metrics


def _mean_psnr(metrics: list[dict[str, object]]) -> float:
    scores = [float(item["psnr"]) for item in metrics if "psnr" in item]
    return float(np.mean(scores)) if scores else float("-inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=None, help="omit for continuous training")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--source-budget", type=int, default=None, help="optional source-size limit; omit for quality-first training")
    parser.add_argument("--length-weight", type=float, default=0.0, help="MDL loss weight; zero prioritises visual/AST accuracy")
    parser.add_argument("--validate-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest_yeganeh_model.pt"))
    parser.add_argument("--best-checkpoint", type=Path, default=Path("checkpoints/best_yeganeh_model.pt"))
    parser.add_argument("--metrics-log", type=Path, default=Path("checkpoints/yeganeh_training.jsonl"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive when supplied")
    if min(args.batch_size, args.accum_steps, args.validate_every) <= 0:
        raise SystemExit("batch, accumulation, and validation intervals must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled = args.amp == "on" or (args.amp == "auto" and device.type in {"cuda", "mps"})
    if amp_enabled and device.type == "cpu":
        raise SystemExit("--amp=on requires CUDA or MPS")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda") if device.type == "cuda" else None

    vocabulary = YeganehVocabulary()
    model = ImageToAST(vocabulary=vocabulary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    step, best_score = 0, float("-inf")
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("format") != "image_to_ast_v1":
            raise SystemExit(f"unsupported checkpoint: {args.resume}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scaler is not None and saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
        step, best_score = int(saved.get("step", 0)), float(saved.get("best_validation_psnr", float("-inf")))

    dataset = SyntheticYeganehDataset(
        count=2**31 - 1, image_size=args.image_size, seed=args.seed,
        vocabulary=vocabulary, max_source_length=args.source_budget,
    )
    args.metrics_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device.type} amp={amp_enabled} batch={args.batch_size} accumulation={args.accum_steps} start_step={step}")
    started = time.perf_counter()
    model.train()
    while args.max_steps is None or step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(args.accum_steps):
            batch_index = (step * args.accum_steps + micro_step) * args.batch_size
            samples = [dataset.sample(batch_index + offset, step=step) for offset in range(args.batch_size)]
            batch = collate_programs(samples, vocabulary)
            image = batch["image"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            target = batch["target"].to(device)
            context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled)
            with context:
                logits = model(image, decoder_input)
                loss = sequence_cross_entropy(logits, target, vocabulary.pad_id, length_weight=args.length_weight) / args.accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated_loss += float(loss.detach().cpu())
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step += 1

        if step == 1 or step % args.log_every == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(f"step={step:07d} phase={curriculum_phase(step)} loss={accumulated_loss:.5f} steps/s={step / elapsed:.2f}")
        if step % args.validate_every != 0:
            continue

        metrics = evaluate_fixed_images(model, args.image_size, DEFAULT_VALIDATION_IMAGES, args.source_budget)
        score = _mean_psnr(metrics)
        improved = score > best_score
        payload = _checkpoint_payload(model, optimizer, scaler, vocabulary, args.image_size, args.seed, step, max(best_score, score))
        _atomic_save(payload, args.checkpoint)
        if improved:
            best_score = score
            _atomic_save(payload, args.best_checkpoint)
        event = {"step": step, "phase": curriculum_phase(step), "loss": accumulated_loss, "mean_psnr": score, "best": improved, "metrics": metrics}
        with args.metrics_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        state = "NEW BEST" if improved else "warning: validation did not improve"
        print(f"validation step={step} mean_PSNR={score:.2f} {state}; latest={args.checkpoint}")


if __name__ == "__main__":
    main()
