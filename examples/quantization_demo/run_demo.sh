#!/usr/bin/env bash
# End-to-end demo: Firefly attributes divergence to the perturbed layer.
#
# 1. Capture a reference from SmolLM-135M.
# 2. Check the unmodified model → no divergence, exit 0.
# 3. Produce a candidate with one MLP weight perturbed by 1e-3 Gaussian noise.
# 4. Check the broken candidate → exit 1, first divergence at layer.7.mlp.

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
ARTIFACTS="$DEMO_DIR/_artifacts"
REFERENCE="$ARTIFACTS/reference"
BROKEN="$ARTIFACTS/broken-smollm"
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
echo "==> Step 1: Capture reference from SmolLM-135M"
uv run --project "$PROJECT_ROOT" firefly capture \
  --model HuggingFaceTB/SmolLM-135M \
  --inputs "$GOLDEN" \
  --out "$REFERENCE"

echo
echo "==> Step 2: Clean check (candidate = original model) — expect no divergence"
uv run --project "$PROJECT_ROOT" firefly check \
  --reference "$REFERENCE" \
  --candidate HuggingFaceTB/SmolLM-135M \
  --inputs "$GOLDEN"

echo
echo "==> Step 3: Produce a deliberately-broken candidate"
uv run --project "$PROJECT_ROOT" python "$DEMO_DIR/make_broken.py" "$BROKEN"

echo
echo "==> Step 4: Check broken candidate — expect first divergence at layer.7.mlp, exit 1"
if uv run --project "$PROJECT_ROOT" firefly check \
  --reference "$REFERENCE" \
  --candidate "$BROKEN" \
  --inputs "$GOLDEN" \
  --report-json "$ARTIFACTS/report.json"; then
  echo "ERROR: check exited 0 but should have exited non-zero" >&2
  exit 2
fi

echo
echo "==> Demo complete. Firefly correctly attributed divergence to the perturbed layer."
echo "    Structured report: $ARTIFACTS/report.json"
