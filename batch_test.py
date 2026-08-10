"""Encode, decode, and visually benchmark ImageToEquation paths in batch.

Each image first passes through the production procedural gates used by
``run_all.py``.  Accepted SDF/multi-scale programs are stored in a compact
``I2ES`` (symbolic-source) container and decoded by evaluating that source.
Rejected images are encoded with the project's reliable block-DCT ``I2E2``
container and decoded with its native decoder.  The report therefore records a
real binary round trip for either path.

Run after collecting fixtures:
    python3 download_test_images.py
    python3 batch_test.py
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys
import time
import zlib

import cv2
import numpy as np
import torch

from evaluator import GridEvaluator
from multiscale_fitter import psnr, ssim
from run_all import _try_multiscale_procedural_path, _try_symbolic_path


ROOT = Path(__file__).resolve().parent
PACKAGE_SRC = ROOT / "ImageToEquation" / "ImageToEquation" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))
from image_to_equation.dct_extractor import extract_dct_descriptors_v2  # noqa: E402
from image_to_equation.i2e_decoder import decode_i2e_to_image  # noqa: E402
from image_to_equation.i2e_encoder import encode_i2e_v2  # noqa: E402


SYMBOLIC_MAGIC = b"I2ES"
SYMBOLIC_VERSION = 1


def encode_symbolic_i2e(equation: str, width: int, height: int, mode: str, path: Path) -> None:
    """Store executable, resolution-independent source in a compressed I2ES file."""
    payload = json.dumps(
        {"version": SYMBOLIC_VERSION, "width": width, "height": height, "mode": mode, "equation": equation},
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(SYMBOLIC_MAGIC + zlib.compress(payload, level=9))


def decode_any_i2e(path: Path) -> tuple[np.ndarray, str]:
    """Decode either the benchmark symbolic I2ES or native block-DCT I2E2 format."""
    magic = path.read_bytes()[:4]
    if magic == SYMBOLIC_MAGIC:
        payload = json.loads(zlib.decompress(path.read_bytes()[4:]).decode("utf-8"))
        evaluator = GridEvaluator()
        rendered = evaluator.evaluate(
            payload["equation"], payload["width"], payload["height"], render_mode="linear"
        )
        return rendered.cpu().numpy(), payload["mode"]
    if magic == b"I2E2":
        return decode_i2e_to_image(str(path)), "dct_i2e2"
    raise ValueError(f"unsupported i2e magic {magic!r}")


def _resize_rgb(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _metrics(reference: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    target = torch.as_tensor(reference, dtype=torch.float32) / 255.0
    render = torch.as_tensor(decoded, dtype=torch.float32) / 255.0
    return psnr(render, target), ssim(render, target)


def _category_for(path: Path) -> str:
    for category in ("structured", "natural", "hybrid"):
        if path.name.startswith(f"{category}_"):
            return category
    return "unlabelled"


def process_one(source: Path, output_dir: Path, max_side: int, dct_quality: int) -> dict[str, object]:
    original_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise ValueError("not a readable raster image")
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    target = _resize_rgb(original, max_side)
    height, width = target.shape[:2]
    evaluator = GridEvaluator()
    started = time.perf_counter()

    result = _try_symbolic_path(target, evaluator)
    if result is None:
        result = _try_multiscale_procedural_path(target, evaluator)

    stem = source.stem
    i2e_path = output_dir / f"{stem}.i2e"
    if result is not None:
        mode = "multiscale_procedural" if "psnr" in result else "symbolic_sdf"
        encode_symbolic_i2e(str(result["equation"]), width, height, mode, i2e_path)
    else:
        mode = "dct_i2e2"
        descriptors = extract_dct_descriptors_v2(target, quality=dct_quality)
        encode_i2e_v2(descriptors, str(i2e_path))

    decoded, decoded_mode = decode_any_i2e(i2e_path)
    if decoded.shape != target.shape:
        raise ValueError(f"decoder shape {decoded.shape} does not match encoded target {target.shape}")
    decoded_path = output_dir / f"decoded_{stem}.png"
    cv2.imwrite(str(decoded_path), cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR))
    psnr_value, ssim_value = _metrics(target, decoded)
    elapsed = time.perf_counter() - started
    return {
        "filename": source.name,
        "category": _category_for(source),
        "path": mode,
        "decoded_mode": decoded_mode,
        "original_dimensions": f"{original.shape[1]}×{original.shape[0]}",
        "encoded_dimensions": f"{width}×{height}",
        "original_bytes": source.stat().st_size,
        "i2e_bytes": i2e_path.stat().st_size,
        "compression_ratio": source.stat().st_size / max(i2e_path.stat().st_size, 1),
        "psnr": psnr_value,
        "ssim": ssim_value,
        "seconds": elapsed,
        "i2e_file": i2e_path.name,
        "decoded_file": decoded_path.name,
    }


def write_report(records: list[dict[str, object]], image_dir: Path, output_dir: Path) -> Path:
    rows = []
    for record in records:
        original_rel = os.path.relpath(image_dir / str(record["filename"]), output_dir)
        decoded_rel = str(record["decoded_file"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['category']))}<br><code>{html.escape(str(record['filename']))}</code></td>"
            f"<td><img src=\"{html.escape(original_rel)}\" alt=\"original\"></td>"
            f"<td><img src=\"{html.escape(decoded_rel)}\" alt=\"decoded\"></td>"
            f"<td><b>{html.escape(str(record['path']))}</b><br>PSNR: {float(record['psnr']):.2f} dB<br>"
            f"SSIM: {float(record['ssim']):.4f}<br>Original: {int(record['original_bytes']):,} B<br>"
            f".i2e: {int(record['i2e_bytes']):,} B<br>Ratio: {float(record['compression_ratio']):.2f}×<br>"
            f"Time: {float(record['seconds']):.2f}s<br><a href=\"{html.escape(str(record['i2e_file']))}\">payload</a></td>"
            "</tr>"
        )
    document = """<!doctype html><html><head><meta charset=\"utf-8\"><title>ImageToEquation benchmark</title>
<style>body{font:14px system-ui;margin:24px;background:#f7f7f7}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #ddd;padding:10px;vertical-align:top}th{background:#20232a;color:white}img{max-width:260px;max-height:220px;object-fit:contain;background:#eee}code{font-size:11px}</style>
</head><body><h1>ImageToEquation batch round-trip benchmark</h1><p>Every row was encoded to a real <code>.i2e</code> payload, decoded, and compared to the actual encoded-resolution RGB input. <code>I2ES</code> holds compressed executable procedural source; <code>I2E2</code> is the native block-DCT fallback.</p>
<table><thead><tr><th>Fixture</th><th>Original</th><th>Decoded</th><th>Round-trip metrics</th></tr></thead><tbody>""" + "\n".join(rows) + "</tbody></table></body></html>\n"
    report = output_dir / "benchmark_report.html"
    report.write_text(document, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "test_images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_output")
    parser.add_argument("--max-side", type=int, default=192, help="maximum encoded side length")
    parser.add_argument("--dct-quality", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None, help="process at most this many unreported fixtures")
    parser.add_argument("--no-resume", action="store_true", help="discard existing benchmark results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        path for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not candidates:
        raise SystemExit(f"no raster fixtures found in {args.image_dir}; run download_test_images.py first")
    results_path = args.output_dir / "results.json"
    records: list[dict[str, object]] = []
    if not args.no_resume and results_path.exists():
        records = json.loads(results_path.read_text(encoding="utf-8"))
    completed = {str(record["filename"]) for record in records}
    processed = 0
    for source in candidates:
        if source.name in completed:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        try:
            record = process_one(source, args.output_dir, args.max_side, args.dct_quality)
            records.append(record)
            print(f"{source.name:40} {record['path']:24} {record['psnr']:.2f} dB {record['ssim']:.4f}")
            write_report(records, args.image_dir, args.output_dir)
            results_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        except Exception as error:
            print(f"FAILED {source.name}: {error}")
        processed += 1
    if not records:
        raise SystemExit("all batch items failed")
    report = write_report(records, args.image_dir, args.output_dir)
    results_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
