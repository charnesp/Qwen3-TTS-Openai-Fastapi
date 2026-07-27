#!/usr/bin/env python3
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Isolate torch.compile / Inductor crashes for the optimized Qwen3-TTS path.

Reproduces the failure mode seen in Docker (BrokenProcessPool, GPU SMEM
autotune warnings, 500 on /v1/audio/speech) without going through the API.

Usage (from repo root, same env as the GHCR image if possible):

    # Full matrix (slow)
    python examples/debug_torch_compile_inductor.py \\
        --ref-audio /path/to/audrey.wav \\
        --ref-text "…"

    # Fast path: diagnostics + inductor probe + baseline only
    python examples/debug_torch_compile_inductor.py --quick

    # Config.yaml-equivalent on Ada (reduce-overhead + codebook compile).
    # Quote the name: '+' is fine in the token; probes are skipped with --only.
    python examples/debug_torch_compile_inductor.py --only 'compile_reduce_overhead+codebook'

    # Force single-threaded inductor (better tracebacks)
    TORCHINDUCTOR_COMPILE_THREADS=1 python examples/debug_torch_compile_inductor.py

Exit codes:
    0  baseline OK (compile may still fail — see printed verdict)
    1  baseline / model load failed
    2  unexpected error
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Must be set before importing torch for compile-thread behavior to apply cleanly.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here.parent / "qwen_tts").exists():
        return here.parent
    cwd = Path.cwd()
    if (cwd / "qwen_tts").exists():
        return cwd
    for p in cwd.resolve().parents:
        if (p / "qwen_tts").exists():
            return p
    return here.parent


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {exc}>"


def gpu_mem(torch_mod) -> dict[str, float]:
    if not torch_mod.cuda.is_available():
        return {}
    torch_mod.cuda.synchronize()
    return {
        "alloc_GiB": round(torch_mod.cuda.memory_allocated() / 1024**3, 3),
        "reserved_GiB": round(torch_mod.cuda.memory_reserved() / 1024**3, 3),
    }


class ResultLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, label: str, ok: bool, torch_mod=None, **kwargs: Any) -> dict[str, Any]:
        row = {"label": label, "ok": ok, **kwargs}
        if torch_mod is not None:
            row.update(gpu_mem(torch_mod))
        self.rows.append(row)
        status = "OK" if ok else "FAIL"
        brief = {k: v for k, v in kwargs.items() if k != "error"}
        print(f"[{status}] {label}: {brief}")
        if not ok and kwargs.get("error"):
            err = str(kwargs["error"])
            print(err if len(err) < 4000 else err[:4000] + "\n…[truncated]")
        return row


def print_diagnostics(torch_mod) -> None:
    print("=== Host ===")
    print("platform:", platform.platform())
    print("python:", sys.version.replace("\n", " "))
    print("torch:", torch_mod.__version__, "cuda:", torch_mod.version.cuda)
    print("cuda available:", torch_mod.cuda.is_available())
    print("TORCHINDUCTOR_COMPILE_THREADS:", os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"))
    print("ROOT:", ROOT)

    if torch_mod.cuda.is_available():
        props = torch_mod.cuda.get_device_properties(0)
        print("\n=== GPU ===")
        print("name:", props.name)
        print("total VRAM GiB:", round(props.total_memory / 1024**3, 2))
        print("SM count:", props.multi_processor_count)
        smem = getattr(props, "shared_memory_per_block", None)
        smem_optin = getattr(props, "shared_memory_per_block_optin", None)
        print("shared_memory_per_block:", smem)
        print("shared_memory_per_block_optin:", smem_optin)
        if smem:
            print(f"→ HW SMEM limit ≈ {smem} bytes (crash log needed 110592)")

    print("\n=== /dev/shm (Docker default often 64M) ===")
    print(sh("df -h /dev/shm"))
    print("\n=== nvidia-smi ===")
    print(
        sh(
            "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free "
            "--format=csv"
        )
    )
    print()


def probe_inductor(torch_mod, mode: str, log: ResultLog) -> None:
    torch_mod.set_float32_matmul_precision("high")
    x = torch_mod.randn(512, 512, device="cuda", dtype=torch_mod.bfloat16)

    @torch_mod.compile(mode=mode, fullgraph=False)
    def f(t):
        return t @ t.T

    t0 = time.perf_counter()
    try:
        y = f(x)
        torch_mod.cuda.synchronize()
        y = f(x)
        torch_mod.cuda.synchronize()
        log.record(
            f"inductor_probe/{mode}",
            True,
            torch_mod,
            seconds=round(time.perf_counter() - t0, 3),
            out_norm=float(y.float().norm()),
        )
    except Exception as exc:  # noqa: BLE001
        log.record(
            f"inductor_probe/{mode}",
            False,
            torch_mod,
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
    finally:
        del x
        gc.collect()
        torch_mod.cuda.empty_cache()


@dataclass
class OptCfg:
    name: str
    use_compile: bool
    compile_mode: str = "default"
    use_cuda_graphs: bool = False
    use_fast_codebook: bool = True
    compile_codebook_predictor: bool = True


MATRIX = [
    OptCfg("no_compile", use_compile=False, compile_codebook_predictor=False),
    OptCfg("compile_default", use_compile=True, compile_mode="default", compile_codebook_predictor=False),
    OptCfg(
        "compile_reduce_overhead",
        use_compile=True,
        compile_mode="reduce-overhead",
        compile_codebook_predictor=False,
    ),
    # Matches a typical production config.yaml on Ada (4080): reduce-overhead +
    # fast codebook + compiled codebook predictor (NOT max-autotune).
    OptCfg(
        "compile_reduce_overhead+codebook",
        use_compile=True,
        compile_mode="reduce-overhead",
        use_cuda_graphs=False,
        use_fast_codebook=True,
        compile_codebook_predictor=True,
    ),
    OptCfg(
        "compile_max_autotune",
        use_compile=True,
        compile_mode="max-autotune",
        compile_codebook_predictor=False,
    ),
    OptCfg(
        "compile_default+codebook",
        use_compile=True,
        compile_mode="default",
        compile_codebook_predictor=True,
    ),
]


def make_ref_audio(ref_audio: Optional[str], torch_mod):
    import numpy as np

    if ref_audio and Path(ref_audio).exists():
        return str(Path(ref_audio)), False
    sr = 24000
    t = np.arange(sr) / sr
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    print("REF_AUDIO missing → synthetic sine, x_vector_only=True")
    return (audio, sr), True


def load_model(model_id: str, device: str, dtype, attn: str, log: ResultLog, torch_mod):
    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn,
        )
        log.record(
            "load_model",
            True,
            torch_mod,
            attn=attn,
            seconds=round(time.perf_counter() - t0, 2),
        )
        return model
    except Exception as exc:  # noqa: BLE001
        log.record("load_model", False, torch_mod, attn=attn, error=str(exc))
        if attn != "sdpa":
            print("Retry with sdpa…")
            return load_model(model_id, device, dtype, "sdpa", log, torch_mod)
        raise


def run_matrix(
    *,
    model_id: str,
    device: str,
    attn: str,
    ref_audio: Optional[str],
    ref_text: str,
    test_text: str,
    language: str,
    decode_window: int,
    emit_every: int,
    configs: list[OptCfg],
    do_stream: bool,
    log: ResultLog,
    torch_mod,
) -> int:
    import numpy as np
    from qwen_tts import Qwen3TTSModel  # noqa: F401

    torch_mod.set_float32_matmul_precision("high")
    dtype = torch_mod.bfloat16

    ref, x_vector_only = make_ref_audio(ref_audio, torch_mod)

    def fresh_model_and_prompt():
        model = load_model(model_id, device, dtype, attn, log, torch_mod)
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref,
            ref_text=None if x_vector_only else ref_text,
            x_vector_only_mode=x_vector_only,
        )
        return model, prompt

    # Baseline without compile
    model, prompt = fresh_model_and_prompt()
    t0 = time.perf_counter()
    try:
        wavs, sr = model.generate_voice_clone(
            text=test_text,
            language=language,
            voice_clone_prompt=prompt,
        )
        dt = time.perf_counter() - t0
        dur = len(wavs[0]) / sr
        log.record(
            "baseline_generate_no_compile",
            True,
            torch_mod,
            seconds=round(dt, 2),
            audio_s=round(dur, 2),
            rtf=round(dt / dur, 3) if dur else None,
        )
    except Exception as exc:  # noqa: BLE001
        log.record(
            "baseline_generate_no_compile",
            False,
            torch_mod,
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        return 1

    for cfg in configs:
        print("\n" + "=" * 72)
        print("CONFIG:", cfg)
        print("=" * 72)

        del model
        gc.collect()
        torch_mod.cuda.empty_cache()
        model, prompt = fresh_model_and_prompt()

        t0 = time.perf_counter()
        try:
            model.enable_streaming_optimizations(
                decode_window_frames=decode_window,
                use_compile=cfg.use_compile,
                use_cuda_graphs=cfg.use_cuda_graphs,
                compile_mode=cfg.compile_mode,
                use_fast_codebook=cfg.use_fast_codebook,
                compile_codebook_predictor=cfg.compile_codebook_predictor,
            )
            log.record(
                f"{cfg.name}/enable",
                True,
                torch_mod,
                seconds=round(time.perf_counter() - t0, 2),
                **asdict(cfg),
            )
        except Exception as exc:  # noqa: BLE001
            log.record(
                f"{cfg.name}/enable",
                False,
                torch_mod,
                error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                **asdict(cfg),
            )
            continue

        t0 = time.perf_counter()
        try:
            model.generate_voice_clone(
                text="Warmup.",
                language="English",
                voice_clone_prompt=prompt,
            )
            log.record(
                f"{cfg.name}/warmup_generate",
                True,
                torch_mod,
                seconds=round(time.perf_counter() - t0, 2),
            )
        except Exception as exc:  # noqa: BLE001
            log.record(
                f"{cfg.name}/warmup_generate",
                False,
                torch_mod,
                error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            continue

        t0 = time.perf_counter()
        try:
            wavs, sr = model.generate_voice_clone(
                text=test_text,
                language=language,
                voice_clone_prompt=prompt,
            )
            dt = time.perf_counter() - t0
            dur = len(wavs[0]) / sr
            log.record(
                f"{cfg.name}/generate",
                True,
                torch_mod,
                seconds=round(dt, 2),
                audio_s=round(dur, 2),
                rtf=round(dt / dur, 3) if dur else None,
            )
        except Exception as exc:  # noqa: BLE001
            log.record(
                f"{cfg.name}/generate",
                False,
                torch_mod,
                error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            continue

        if not do_stream:
            continue

        t0 = time.perf_counter()
        first = None
        chunks: list = []
        try:
            for chunk, chunk_sr in model.stream_generate_voice_clone(
                text=test_text,
                language=language,
                voice_clone_prompt=prompt,
                emit_every_frames=emit_every,
                decode_window_frames=decode_window,
            ):
                if first is None:
                    first = time.perf_counter() - t0
                chunks.append(chunk)
                sr = chunk_sr
            dt = time.perf_counter() - t0
            audio = np.concatenate(chunks) if chunks else np.array([])
            dur = len(audio) / sr if sr else 0
            log.record(
                f"{cfg.name}/stream",
                True,
                torch_mod,
                seconds=round(dt, 2),
                ttfb_s=round(first, 3) if first else None,
                chunks=len(chunks),
                audio_s=round(dur, 2),
                rtf=round(dt / dur, 3) if dur else None,
            )
        except Exception as exc:  # noqa: BLE001
            log.record(
                f"{cfg.name}/stream",
                False,
                torch_mod,
                error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )

    return 0


def print_verdict(log: ResultLog) -> None:
    print("\n=== Verdict ===")
    baseline_ok = any(
        r["label"] == "baseline_generate_no_compile" and r["ok"] for r in log.rows
    )
    probe_fail = [
        r for r in log.rows if r["label"].startswith("inductor_probe/") and not r["ok"]
    ]
    opt_fail = [
        r
        for r in log.rows
        if (("/enable" in r["label"] or "/generate" in r["label"] or "/stream" in r["label"])
            and not r["ok"])
    ]

    if not baseline_ok:
        print("Baseline sans compile KO → modèle / FA2 / weights, pas Inductor.")
    elif probe_fail:
        print(
            "Mini-repro Inductor KO → Inductor/GPU SMEM ou /dev/shm Docker. "
            "Contournement: use_compile=false ou shm_size=2gb."
        )
        for r in probe_fail:
            print(" -", r["label"])
    elif opt_fail:
        print("Baseline OK, mais optimizations cassent:")
        for r in opt_fail:
            print(" -", r["label"])
        print("→ garde le mode le plus agressif encore OK.")
    else:
        print("Tout OK ici — si l’API crash encore, vérifier le YAML (défaut max-autotune).")

    good = [
        r
        for r in log.rows
        if r["ok"] and r["label"].endswith("/generate") and "no_compile" not in r["label"]
    ]
    if good:
        best = min(good, key=lambda r: r.get("rtf") or 999)
        print("Meilleure generate compilée:", best["label"], "RTF", best.get("rtf"))

    print("\n=== Results (compact) ===")
    for r in log.rows:
        mark = "✓" if r["ok"] else "✗"
        extra = []
        for k in ("seconds", "rtf", "ttfb_s", "attn", "compile_mode"):
            if k in r and r[k] is not None:
                extra.append(f"{k}={r[k]}")
        print(f"  {mark} {r['label']}" + (f"  ({', '.join(extra)})" if extra else ""))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attn", default="flash_attention_2", help="flash_attention_2|sdpa|eager")
    p.add_argument("--ref-audio", default=None, help="WAV for ICL clone; synthetic sine if omitted")
    p.add_argument(
        "--ref-text",
        default="Bonjour, ceci est un échantillon de référence pour le clonage vocal.",
    )
    p.add_argument(
        "--text",
        default="Bonjour, ceci est un test de synthèse vocale pour diagnostiquer torch.compile.",
    )
    p.add_argument("--language", default="French")
    p.add_argument("--decode-window", type=int, default=80)
    p.add_argument("--emit-every", type=int, default=6)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Diagnostics + inductor probes + baseline only (skip optimization matrix)",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Skip streaming tests in the matrix",
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Run only named matrix config(s). Repeatable. "
            "Example: --only 'compile_reduce_overhead+codebook'. "
            "Implies --skip-probes (max-autotune probe segfaults on some Ada GPUs)."
        ),
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Write results JSON (default: examples/debug_compile_results_<ts>.json)",
    )
    p.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip tiny inductor matmul probes",
    )
    p.add_argument(
        "--probe-max-autotune",
        action="store_true",
        help=(
            "Include the max-autotune inductor probe (often SIGSEGV on RTX 40xx "
            "+ torch 2.5 / Triton; off by default)"
        ),
    )
    p.add_argument(
        "--diagnose-stream",
        action="store_true",
        help=(
            "Isolate why streaming segfaults with reduce-overhead: "
            "non-stream uses decoder.chunked_decode (eager), stream uses "
            "forward_optimized (compiled + CUDA graphs). Runs progressive tests."
        ),
    )
    return p.parse_args(argv)


def diagnose_stream_crash(args, log: ResultLog, torch_mod) -> int:
    """Progressive isolation of the streaming + reduce-overhead segfault."""
    import faulthandler
    import numpy as np

    faulthandler.enable()
    torch_mod.set_float32_matmul_precision("high")

    print(
        "\n=== HYPOTHESIS ===\n"
        "Non-stream generate() calls speech_tokenizer.decode → decoder.chunked_decode\n"
        "  → decoder.forward()  (EAGER — ignores torch.compile)\n"
        "Stream calls speech_tokenizer.decode_streaming → decode_padded\n"
        "  → decoder.forward_optimized() → _compiled_forward  (Inductor CUDA graphs)\n"
        "So generate OK + stream segfault means the compiled decoder path is the crash.\n"
    )

    ref, x_vector_only = make_ref_audio(args.ref_audio, torch_mod)
    model = load_model(
        args.model_id, args.device, torch_mod.bfloat16, args.attn, log, torch_mod
    )
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref,
        ref_text=None if x_vector_only else args.ref_text,
        x_vector_only_mode=x_vector_only,
    )

    print("\n--- enable reduce-overhead (decoder compile, no codebook compile) ---")
    model.enable_streaming_optimizations(
        decode_window_frames=args.decode_window,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="reduce-overhead",
        use_fast_codebook=True,
        compile_codebook_predictor=False,
    )

    tok = model.model.speech_tokenizer
    # Wrapper (Qwen3TTSTokenizer) exposes .model = HF tokenizer module
    inner = getattr(tok, "model", tok)
    decoder = inner.decoder
    has_compiled = getattr(decoder, "_compiled_forward", None) is not None
    print(f"decoder._compiled_forward set: {has_compiled}")
    print(f"decoder._compile_mode: {getattr(decoder, '_compile_mode', None)}")
    print(
        "NOTE: generate() → chunked_decode → decoder.forward()  "
        "(does NOT call _compiled_forward)"
    )

    # A) Non-stream (known OK) — proves eager path
    print("\n--- A) non-stream generate (uses chunked_decode / eager) ---")
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=args.text, language=args.language, voice_clone_prompt=prompt
    )
    log.record(
        "diagnose/A_generate_eager_path",
        True,
        torch_mod,
        seconds=round(time.perf_counter() - t0, 2),
        samples=len(wavs[0]),
    )

    # B) Isolated compiled decode, fixed pad size, N times
    print("\n--- B) decoder.forward_optimized loop (first real use of compile) ---")
    Q = decoder.config.num_quantizers
    T = args.decode_window
    device = next(decoder.parameters()).device
    codes = torch_mod.randint(0, 1024, (1, Q, T), device=device, dtype=torch_mod.long)
    try:
        for i in range(8):
            torch_mod.compiler.cudagraph_mark_step_begin()
            out = decoder.forward_optimized(codes)
            torch_mod.cuda.synchronize()
            print(f"  forward_optimized #{i+1}: out={tuple(out.shape)}")
        log.record("diagnose/B_forward_optimized_x8", True, torch_mod)
    except Exception as exc:  # noqa: BLE001
        log.record(
            "diagnose/B_forward_optimized_x8",
            False,
            torch_mod,
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        print("Crashed/exception in B — compiled decoder is the culprit.")
        print_verdict(log)
        return 1

    # C) decode_streaming API (same as stream_generate uses)
    print("\n--- C) speech_tokenizer.decode_streaming x8 (pad_to_size) ---")
    # [T, Q] window like stream_generate
    window = torch_mod.randint(0, 1024, (T, Q), device=device, dtype=torch_mod.long)
    try:
        for i in range(8):
            wav_list, _sr = tok.decode_streaming(
                window, use_optimized=True, pad_to_size=T
            )
            print(f"  decode_streaming #{i+1}: samples={len(wav_list[0])}")
        log.record("diagnose/C_decode_streaming_x8", True, torch_mod)
    except Exception as exc:  # noqa: BLE001
        log.record(
            "diagnose/C_decode_streaming_x8",
            False,
            torch_mod,
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        print("Crashed in C — decode_streaming/pad path.")
        print_verdict(log)
        return 1

    # D) Full stream with optimized decode OFF (should survive if hypothesis holds)
    print("\n--- D) stream_generate use_optimized_decode=False ---")
    try:
        n = 0
        t0 = time.perf_counter()
        for chunk, _sr in model.stream_generate_voice_clone(
            text=args.text,
            language=args.language,
            voice_clone_prompt=prompt,
            emit_every_frames=args.emit_every,
            decode_window_frames=args.decode_window,
            use_optimized_decode=False,
        ):
            n += 1
        log.record(
            "diagnose/D_stream_unoptimized_decode",
            True,
            torch_mod,
            seconds=round(time.perf_counter() - t0, 2),
            chunks=n,
        )
    except Exception as exc:  # noqa: BLE001
        log.record(
            "diagnose/D_stream_unoptimized_decode",
            False,
            torch_mod,
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    # E) Full stream with optimized decode ON (expected segfault point)
    print("\n--- E) stream_generate use_optimized_decode=True (expected crash point) ---")
    print("If the process dies here with SIGSEGV, hypothesis is confirmed.")
    try:
        n = 0
        t0 = time.perf_counter()
        for chunk, _sr in model.stream_generate_voice_clone(
            text=args.text,
            language=args.language,
            voice_clone_prompt=prompt,
            emit_every_frames=args.emit_every,
            decode_window_frames=args.decode_window,
            use_optimized_decode=True,
        ):
            n += 1
            if n == 1:
                print(f"  first chunk ok, samples={len(chunk)}")
        log.record(
            "diagnose/E_stream_optimized_decode",
            True,
            torch_mod,
            seconds=round(time.perf_counter() - t0, 2),
            chunks=n,
        )
    except Exception as exc:  # noqa: BLE001
        log.record(
            "diagnose/E_stream_optimized_decode",
            False,
            torch_mod,
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    print("\n=== Interpretation ===")
    print("A OK + B/C crash  → compiled decoder alone is enough to die")
    print("A/B/C/D OK + E crash → interaction stream loop + optimized decode")
    print("D OK + E crash     → workaround: stream with use_optimized_decode=False")
    print("                      or use_compile=false in config.yaml")
    print_verdict(log)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = ResultLog()

    try:
        import torch
    except ImportError:
        print("torch is required", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("CUDA is required for this debug script", file=sys.stderr)
        return 2

    print_diagnostics(torch)

    if args.diagnose_stream:
        code = diagnose_stream_crash(args, log, torch)
        _write_json(args, log, torch)
        return code

    # --only targets a TTS matrix config; don't die on the max-autotune probe first.
    skip_probes = args.skip_probes or bool(args.only)
    if args.only and not args.skip_probes:
        print("Note: --only implies --skip-probes (use probes without --only).\n")

    if not skip_probes:
        print("=== Inductor probes (outside TTS) ===")
        probe_modes = ["default", "reduce-overhead"]
        if args.probe_max_autotune:
            probe_modes.append("max-autotune")
        else:
            print("(skipping max-autotune probe; pass --probe-max-autotune to force it)\n")
        for mode in probe_modes:
            print(f"\n--- probe {mode} ---")
            probe_inductor(torch, mode, log)

    if args.quick:
        # Still run baseline via matrix with empty opt list trick
        code = run_matrix(
            model_id=args.model_id,
            device=args.device,
            attn=args.attn,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            test_text=args.text,
            language=args.language,
            decode_window=args.decode_window,
            emit_every=args.emit_every,
            configs=[],  # baseline only
            do_stream=False,
            log=log,
            torch_mod=torch,
        )
        print_verdict(log)
        _write_json(args, log, torch)
        return code

    configs = MATRIX
    if args.only:
        wanted = set(args.only)
        configs = [c for c in MATRIX if c.name in wanted]
        missing = wanted - {c.name for c in configs}
        if missing:
            print("Unknown --only names:", ", ".join(sorted(missing)), file=sys.stderr)
            print("Available:", ", ".join(c.name for c in MATRIX), file=sys.stderr)
            return 2

    code = run_matrix(
        model_id=args.model_id,
        device=args.device,
        attn=args.attn,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        test_text=args.text,
        language=args.language,
        decode_window=args.decode_window,
        emit_every=args.emit_every,
        configs=configs,
        do_stream=not args.no_stream,
        log=log,
        torch_mod=torch,
    )
    print_verdict(log)
    _write_json(args, log, torch)
    return code


def _write_json(args: argparse.Namespace, log: ResultLog, torch_mod) -> None:
    out = args.json_out
    if out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = str(ROOT / "examples" / f"debug_compile_results_{ts}.json")
    payload = {
        "model_id": args.model_id,
        "gpu": torch_mod.cuda.get_device_name(0) if torch_mod.cuda.is_available() else None,
        "torch": torch_mod.__version__,
        "env": {
            "TORCHINDUCTOR_COMPILE_THREADS": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        },
        "results": log.rows,
    }
    Path(out).write_text(json.dumps(payload, indent=2, default=str))
    print("\nWrote", out)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)
