"""LLM proposer — the agent that searches the recipe space, grounded by Firefly.

The harness's ``propose(bundle, history) -> action`` slot takes any callable; the
deterministic router is one impl, this is the reference Anthropic tool-use impl.
The LLM emits a **compact, unit-level action** (not raw FQNs) via a forced tool
call, which the harness expands to a :class:`Recipe` and verifies — so the action
is sandboxed (only validated interventions, never code) and every proposal is
measured. A hallucinated action is just a valid recipe that the gate rejects.
"""

from __future__ import annotations

import json

from firefly.quant.awq import AWQQuantizer
from firefly.quant.intervention import RTNQuantizer
from firefly.quant.recipe_io import Recipe, build_recipe
from firefly.quant.smoothquant import SmoothQuant

DEFAULT_MODEL = "claude-sonnet-4-6"

#: The structured action the model is forced to emit (Anthropic tool schema).
POLICY_TOOL = {
    "name": "propose_recipe",
    "description": (
        "Propose the next quantization recipe to try, to clear the perplexity bar "
        "at minimum memory. The recipe quantizes all decoder Linears with the chosen "
        "quantizer except units listed in keep_fp_units (kept full precision)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "quantizer": {
                "type": "string", "enum": ["rtn", "awq"],
                "description": "rtn = plain round-to-nearest; awq = activation-aware "
                "(protects salient weight channels, best for int4). awq costs the same bits.",
            },
            "pre_transforms": {
                "type": "array", "items": {"type": "string", "enum": ["smoothquant"]},
                "description": "Pre-transforms before quantizing. smoothquant helps w8a8 "
                "activation outliers; irrelevant (no-op) for weight-only int4.",
            },
            "keep_fp_units": {
                "type": "array", "items": {"type": "string"},
                "description": "Decoder-layer unit names to keep in full precision (e.g. "
                "'layer.24'). Each costs extra memory; keep as few as possible — target the "
                "layers with the highest residual divergence from the previous attempt.",
            },
            "group_size": {"type": "integer", "description": "Quant group size (e.g. 128)."},
            "rationale": {"type": "string", "description": "One sentence: why this recipe."},
        },
        "required": ["quantizer", "keep_fp_units", "rationale"],
    },
}


def compact_to_recipe(
    action: dict,
    *,
    model_id: str,
    scheme: str,
    all_fqns: set[str],
    unit_fqns: dict[str, list[str]],
    inputs_path,
    default_group_size: int = 128,
    dtype: str = "float32",
    device: str = "cpu",
) -> Recipe:
    """Expand a compact unit-level action into a full :class:`Recipe`. Unknown
    unit names are ignored (validated against ``unit_fqns``), so the action stays
    sandboxed."""
    keep_fp = {f for u in action.get("keep_fp_units", []) for f in unit_fqns.get(u, [])}
    group_size = int(action.get("group_size") or default_group_size)
    quantizer = (
        AWQQuantizer(group_size=group_size) if action.get("quantizer") == "awq" else RTNQuantizer()
    )
    pre_transforms = [SmoothQuant()] if "smoothquant" in action.get("pre_transforms", []) else []
    return build_recipe(
        model_id=model_id, scheme=scheme, group_size=group_size, granularity="layer",
        quantize_fqns=set(all_fqns) - keep_fp, kept_fp_fqns=keep_fp,
        pre_transforms=pre_transforms, quantizer=quantizer,
        dtype=dtype, device=device, inputs_path=inputs_path,
        result={"rationale": action.get("rationale", "")},
    )


def parse_tool_action(message) -> dict:
    """Pull the ``propose_recipe`` tool input out of an Anthropic response."""
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "propose_recipe":
            return dict(block.input)
    raise ValueError("model did not call propose_recipe")


def build_prompt(bundle: dict, history: list[dict]) -> str:
    """Render the measurement bundle + attempt history into the user prompt."""
    return (
        "You are tuning a post-training quantization recipe. GOAL: clear the "
        "perplexity bar at MINIMUM weight memory.\n\n"
        "How recipes work: all decoder Linears are quantized with your chosen "
        "quantizer (rtn or awq) at the target scheme, except units you list in "
        "keep_fp_units (kept fp — better quality, more memory). Keep as few fp as "
        "possible; when over the bar, keep the layers with the highest residual "
        "divergence (from the last attempt's attribution).\n\n"
        f"MEASUREMENTS:\n{json.dumps(bundle, indent=2)}\n\n"
        f"ATTEMPTS SO FAR:\n{json.dumps(history, indent=2) if history else '(none yet)'}\n\n"
        "Call propose_recipe with your next attempt."
    )


def propose_with_llm(bundle: dict, history: list[dict], *, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> dict:
    """Reference proposer: one forced tool call → a compact action dict."""
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=[POLICY_TOOL],
        tool_choice={"type": "tool", "name": "propose_recipe"},
        messages=[{"role": "user", "content": build_prompt(bundle, history)}],
    )
    return parse_tool_action(msg)
