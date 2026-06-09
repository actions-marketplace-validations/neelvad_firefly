#!/usr/bin/env bash
# End-to-end per-head demo: Firefly attributes divergence to the exact
# attention head, not just the layer.
#
# 1. Capture a reference from SmolLM-135M WITH --per-head (adds attn_heads taps).
# 2. Check the unmodified model → no divergence, exit 0.
# 3. Produce a candidate with ONE query head (layer 7, head 4) perturbed.
# 4. Check the broken candidate → exit 1. The per-head attribution table names
#    head 4 of layer.7.attn_heads as the worst head, with high concentration
#    (worst / median head) since the divergence is localized to that head.

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
ARTIFACTS="$DEMO_DIR/_artifacts"
REFERENCE="$ARTIFACTS/reference"
BROKEN="$ARTIFACTS/broken-head-smollm"
GOLDEN="$ARTIFACTS/golden.json"

mkdir -p "$ARTIFACTS"

echo "==> Writing golden inputs"
cat >"$GOLDEN" <<'EOF'
{
  "texts": ["the quick brown fox jumps over the lazy dog", "to be or not to be"],
  "max_length": 16
}
EOF

echo
echo "==> Step 1: Capture reference from SmolLM-135M (--per-head)"
uv run --project "$PROJECT_ROOT" firefly capture \
  --model HuggingFaceTB/SmolLM-135M \
  --inputs "$GOLDEN" \
  --out "$REFERENCE" \
  --per-head

echo
echo "==> Step 2: Clean check (candidate = original model) — expect no divergence"
uv run --project "$PROJECT_ROOT" firefly check \
  --reference "$REFERENCE" \
  --candidate HuggingFaceTB/SmolLM-135M \
  --inputs "$GOLDEN" \
  --allow-default-tolerances

echo
echo "==> Step 3: Produce a candidate with one query head perturbed"
uv run --project "$PROJECT_ROOT" python "$DEMO_DIR/make_broken_head.py" "$BROKEN"

echo
echo "==> Step 4: Check broken candidate — expect exit 1 + per-head table naming head 4"
if uv run --project "$PROJECT_ROOT" firefly check \
  --reference "$REFERENCE" \
  --candidate "$BROKEN" \
  --inputs "$GOLDEN" \
  --allow-default-tolerances \
  --report-json "$ARTIFACTS/report.json"; then
  echo "ERROR: check exited 0 but should have exited non-zero" >&2
  exit 2
fi

echo
echo "==> Demo complete. The per-head attribution table localized divergence to a single head."
echo "    Structured report (includes per_head with worst_head + concentration): $ARTIFACTS/report.json"
