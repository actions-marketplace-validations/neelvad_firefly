"""Reconnaissance: load a HF model in vLLM, inspect its internal module tree.

This is the *first* step toward a vLLM tap-point selector in Firefly. It
deliberately does no capture and no diff — its only job is to print the
underlying ``nn.Module`` structure so we can decide which tap names map
cleanly across the HF-transformers reference and the vLLM candidate.

Usage:
    uv run modal run scripts/vllm_explore.py
    uv run modal run scripts/vllm_explore.py --gpu A10G --model HuggingFaceTB/SmolLM-135M
"""

from __future__ import annotations

import os

import modal

app = modal.App("firefly-vllm-explore")

# vLLM bundles its own torch + cuda deps; let pip resolve. We pin to a known
# stable major and let modal cache the image once it builds. Build time is
# slow first run (~5min) because vllm pulls a lot of CUDA libs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.8.5",
        "transformers==4.51.3",
        "huggingface_hub>=0.24",
    )
)

_HF_TOKEN_SET = bool(os.environ.get("HF_TOKEN"))
_HF_SECRETS = (
    [modal.Secret.from_local_environ(["HF_TOKEN"])] if _HF_TOKEN_SET else []
)


def _inspect_model(model) -> dict:
    """Runs inside the vLLM worker process via ``LLM.apply_model``.

    Must be a top-level function so it survives serialization to the worker.
    Returns the module tree + class name; caller assembles the rest.
    """
    return {
        "class": type(model).__name__,
        "modules": [
            {
                "name": name,
                "type": type(m).__name__,
                "n_params_direct": sum(p.numel() for p in m.parameters(recurse=False)),
            }
            for name, m in model.named_modules()
        ],
    }


def _find_underlying_model(llm, max_depth: int = 8) -> tuple[object | None, str | None]:
    """vLLM's path to the underlying nn.Module has changed across versions.

    BFS from the LLM object, looking for an ``nn.Module`` that has a
    ``layers`` attribute that is iterable — that's the decoder stack. Bounded
    depth keeps us safe from cycles.
    """
    import torch.nn as nn

    visited: set[int] = set()
    queue: list[tuple[object, str, int]] = [(llm, "", 0)]
    candidates: list[tuple[object, str]] = []

    while queue:
        obj, path, depth = queue.pop(0)
        if id(obj) in visited or depth > max_depth:
            continue
        visited.add(id(obj))

        # Score: an nn.Module that has .layers (a ModuleList of decoder blocks)
        # OR has .model.layers (wrapping pattern: LlamaForCausalLM has .model
        # which is LlamaModel which has .layers).
        if isinstance(obj, nn.Module):
            inner = getattr(obj, "model", None) if hasattr(obj, "model") else None
            layers_target = inner if (inner is not None and hasattr(inner, "layers")) else obj
            if hasattr(layers_target, "layers"):
                try:
                    if len(layers_target.layers) >= 1:
                        candidates.append((obj, path))
                except (TypeError, AttributeError):
                    pass

        # Expand children, but skip dunder, callables, primitives.
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            if child is None or isinstance(child, (str, int, float, bool, bytes)):
                continue
            if callable(child) and not isinstance(child, nn.Module):
                continue
            queue.append((child, f"{path}.{name}" if path else name, depth + 1))

    if not candidates:
        return None, None
    # Prefer the candidate at the shortest path (most "canonical" handle).
    candidates.sort(key=lambda c: c[1].count("."))
    return candidates[0]


@app.function(gpu="A10G", image=image, timeout=900, secrets=_HF_SECRETS)
def explore_vllm_model(
    model_id: str = "HuggingFaceTB/SmolLM-135M",
    prompt: str = "the quick brown fox",
) -> dict:
    """Load model via vLLM, walk module tree, run one generation as sanity check."""
    # Force V0 engine: V1 (default in 0.8.5) has a broken `apply_model` that
    # references an old executor path. V0 still works for recon; we may
    # revisit collective_rpc on V1 once we move past inspection.
    os.environ["VLLM_USE_V1"] = "0"

    import vllm
    from vllm import LLM, SamplingParams

    print(f"vllm version: {vllm.__version__}  VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")

    # `enforce_eager=True` skips CUDA graph capture so the module tree we see
    # matches the actual forward-pass modules (graph capture wraps things).
    # `max_model_len` small to fit on A10G with this tiny model.
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=0.4,
    )

    top_level_attrs = [a for a in dir(llm) if not a.startswith("_")]
    print("\n=== vLLM engine attribute tree (top level) ===")
    for attr in top_level_attrs:
        print(f"  llm.{attr}")

    # vLLM V1 engine (0.8+) runs the model in a worker process. The canonical
    # way to inspect / hook it from the client is `apply_model(fn)`, which
    # sends a callable to the worker, runs it against the real nn.Module, and
    # returns the result. We try that first; only fall back to BFS for V0.
    module_tree: list[dict] | None = None
    underlying_class: str | None = None
    found_path: str | None = None

    if hasattr(llm, "apply_model"):
        print("\n=== Using V1 engine path: llm.apply_model(_inspect_model) ===")
        result = llm.apply_model(_inspect_model)
        # apply_model returns a list (one entry per tensor-parallel worker);
        # for TP=1 we take the head.
        payload = result[0] if isinstance(result, list) and result else result
        module_tree = payload.get("modules")
        underlying_class = payload.get("class")
        found_path = "apply_model"
        print(f"  underlying class: {underlying_class}")
    else:
        underlying, found_path = _find_underlying_model(llm)
        if underlying is not None:
            print(f"\n=== V0 engine: found model at llm.{found_path} ===")
            underlying_class = type(underlying).__name__
            module_tree = [
                {
                    "name": name,
                    "type": type(m).__name__,
                    "n_params_direct": sum(p.numel() for p in m.parameters(recurse=False)),
                }
                for name, m in underlying.named_modules()
            ]

    if module_tree is None:
        print("\n!!! Could not access nn.Module via apply_model or BFS.")
        return {
            "vllm_version": vllm.__version__,
            "top_level_attrs": top_level_attrs,
            "module_tree": None,
        }

    print(f"\n=== Module tree ({len(module_tree)} modules total) ===")
    for entry in module_tree:
        indent = "  " * entry["name"].count(".")
        param_tag = f" [params={entry['n_params_direct']}]" if entry["n_params_direct"] else ""
        print(f"{indent}{entry['name'] or '<root>'}: {entry['type']}{param_tag}")

    # Sanity: actually run a generation to confirm the model is wired up.
    print("\n=== Sanity generation ===")
    params = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate([prompt], params)
    completion = outputs[0].outputs[0].text
    print(f"  prompt:     {prompt!r}")
    print(f"  completion: {completion!r}")

    return {
        "vllm_version": vllm.__version__,
        "model_id": model_id,
        "underlying_model_path": found_path,
        "underlying_model_class": underlying_class,
        "module_tree": module_tree,
        "sanity_completion": completion,
    }


@app.local_entrypoint()
def main(
    model: str = "HuggingFaceTB/SmolLM-135M",
    gpu: str = "A10G",
    prompt: str = "the quick brown fox",
) -> None:
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    if _HF_TOKEN_SET:
        print("HF_TOKEN found in local env — forwarding to GPU container.")
    print(f"Launching {gpu} job for vLLM model={model}")

    result = explore_vllm_model.with_options(gpu=gpu).remote(
        model_id=model, prompt=prompt,
    )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"vllm_explore_{gpu.lower().replace('-', '_')}_{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"\nResults written to {out_path}")
    if result.get("module_tree"):
        print(f"Underlying model: {result['underlying_model_class']} at llm.{result['underlying_model_path']}")
        print(f"Module count: {len(result['module_tree'])}")
