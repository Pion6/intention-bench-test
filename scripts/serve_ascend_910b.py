#!/usr/bin/env python3
"""One-click Huawei Ascend 910B (torch_npu) System One HTTP server launcher for vllm-sr/d3* models.

Usage on Ascend 910B3 (64 GB HBM):
    # Optional: use domestic Hugging Face mirror if needed
    export HF_ENDPOINT=https://hf-mirror.com

    # Launch flagship 27B d3-ultra on NPU 0 (listening on 0.0.0.0:8000)
    python3 scripts/serve_ascend_910b.py --model vllm-sr/d3-ultra --device npu:0 --host 0.0.0.0 --port 8000

    # Or launch 9B d3-flash / 4B d3-mini
    python3 scripts/serve_ascend_910b.py --model vllm-sr/d3-flash --device npu:0 --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve vllm-sr/d3* (d3-ultra, d3-turbo, d3-flash, d3-mini, etc.) on Huawei Ascend 910B NPU."
    )
    parser.add_argument(
        "--model",
        default="vllm-sr/d3-ultra",
        help="Hugging Face repo ID or local snapshot directory (default: vllm-sr/d3-ultra)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face git revision (branch, tag, or commit)",
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Ascend NPU device string (default: npu:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Questions per forward pass (default: 8)",
    )
    parser.add_argument(
        "--verify",
        choices=("fast", "full", "release", "none"),
        default="fast",
        help="Manifest verification mode (default: fast)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup forward passes on startup",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()

    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is not installed in the current Python environment. "
            "Run inside an Ascend PyTorch container/environment."
        ) from exc

    if args.device.startswith("npu"):
        if not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is False. Check npu-smi info and container NPU mounts.")
        torch.npu.set_device(torch.device(args.device))
        props = torch.npu.get_device_properties(torch.device(args.device))
        total_gb = getattr(props, "total_memory", 0) / (1024**3)
        print(
            f"[Ascend NPU] Using {args.device}: {getattr(props, 'name', '910B')} "
            f"(Total HBM: {total_gb:.2f} GiB)",
            flush=True,
        )

    local_path = Path(args.model)
    if local_path.is_dir():
        snapshot_dir = local_path.resolve()
    else:
        from huggingface_hub import snapshot_download

        print(f"[Ascend NPU] Downloading / resolving {args.model} to local cache...", flush=True)
        snapshot_dir = Path(
            snapshot_download(repo_id=args.model, revision=args.revision)
        ).resolve()

    sys.path.insert(0, str(snapshot_dir))

    # Import d3_runtime and patch D3._sync to also synchronize NPU for accurate timing
    import d3_runtime

    orig_sync = d3_runtime.D3._sync

    def _npu_aware_sync(self) -> None:
        if self.device.type == "npu":
            self.torch.npu.synchronize(self.device)
        else:
            orig_sync(self)

    d3_runtime.D3._sync = _npu_aware_sync

    # Load and delegate to official d3_server.py from the model snapshot
    server_file = snapshot_dir / "d3_server.py"
    spec = importlib.util.spec_from_file_location("d3_server", str(server_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load d3_server.py from {server_file}")
    d3_server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d3_server)

    argv = [
        "--model",
        str(snapshot_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--verify",
        args.verify,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.revision:
        argv.extend(["--revision", args.revision])
    if args.no_warmup:
        argv.append("--no-warmup")

    return d3_server.main(argv)


if __name__ == "__main__":
    sys.exit(main())
