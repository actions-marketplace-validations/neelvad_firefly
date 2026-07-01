"""Is quantization fragility component-specific in a (heterogeneous) recsys model?

The probe that decides whether recsys is the real home for per-layer/per-component
mixed precision. On a homogeneous LLM, fragility smears across ~identical blocks
and a single recovery method handles it (per-layer was moot at 7B). A recsys model
is heterogeneous — big/small embedding tables, cross layers, a deep MLP — so the
hypothesis is that fragility CONCENTRATES in specific components, and no uniform
precision fits all of them (you must measure per-component). This tests it end to
end: train a DCN-v2 on MovieLens-1M (real data, real AUC), then fake-quantize each
component to int4 in isolation and read the AUC drop per component.

Read: if one/two components dominate the AUC loss (and others are ~free), recsys
needs per-component precision allocation — the attribution moat with a problem
that actually needs it. If the loss is even across components, it behaves like the
LLM and per-component is no more useful here.

Standalone (no firefly/torchao/vllm) — a hypothesis probe, not a feature yet.

Run:  uv run modal run scripts/recsys_precision_fragility.py
"""

from __future__ import annotations

import modal

app = modal.App("firefly-recsys-precision-fragility")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "numpy", "scikit-learn", "requests")
)

GPU = "A10G"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@app.function(image=image, gpu=GPU, timeout=3600)
def run() -> dict:
    import copy
    import io
    import zipfile

    import numpy as np
    import requests
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"

    # --- data: MovieLens-1M → binary CTR (rating >= 4) with user/movie + side feats ---
    print("downloading MovieLens-1M ...")
    z = zipfile.ZipFile(io.BytesIO(requests.get(ML1M_URL, timeout=120).content))

    def read(name):
        return z.read(f"ml-1m/{name}").decode("latin-1").strip().split("\n")

    users = {}
    for line in read("users.dat"):
        uid, gender, age, occ, _zip = line.split("::")
        users[int(uid)] = (0 if gender == "M" else 1, int(age), int(occ))
    AGES = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
              "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
              "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    GI = {g: i for i, g in enumerate(GENRES)}
    movie_genres = {}
    for line in read("movies.dat"):
        mid, _title, genres = line.split("::")
        movie_genres[int(mid)] = [GI[g] for g in genres.split("|") if g in GI] or [0]

    rows = []
    for line in read("ratings.dat"):
        uid, mid, rating, _ts = line.split("::")
        uid, mid, rating = int(uid), int(mid), int(rating)
        g, age, occ = users[uid]
        gid = movie_genres.get(mid, [0])
        rows.append((uid, mid, g, AGES[age], occ, gid, 1 if rating >= 4 else 0))
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    n_val = len(rows) // 10
    train, val = rows[n_val:], rows[:n_val]
    print(f"{len(rows)} ratings  ({sum(r[-1] for r in rows) / len(rows):.1%} positive)  "
          f"train {len(train)} / val {len(val)}")

    N_USER, N_MOVIE, N_GENRE, N_OCC = 6041, 3953, 18, 21
    D = 16

    def batchify(data, bs=4096):
        for i in range(0, len(data), bs):
            chunk = data[i:i + bs]
            u = torch.tensor([r[0] for r in chunk], device=dev)
            m = torch.tensor([r[1] for r in chunk], device=dev)
            g = torch.tensor([r[2] for r in chunk], device=dev)
            a = torch.tensor([r[3] for r in chunk], device=dev)
            o = torch.tensor([r[4] for r in chunk], device=dev)
            # genre multi-hot pooled via offsets
            gids, offs, off = [], [], 0
            for r in chunk:
                offs.append(off)
                gids.extend(r[5])
                off += len(r[5])
            gi = torch.tensor(gids, device=dev)
            go = torch.tensor(offs, device=dev)
            y = torch.tensor([r[6] for r in chunk], device=dev, dtype=torch.float32)
            yield u, m, g, a, o, gi, go, y

    class DCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(N_USER, D)
            self.movie_emb = nn.Embedding(N_MOVIE, D)
            self.gender_emb = nn.Embedding(2, D)
            self.age_emb = nn.Embedding(7, D)
            self.occ_emb = nn.Embedding(N_OCC, D)
            self.genre_emb = nn.EmbeddingBag(N_GENRE, D, mode="mean")
            din = 6 * D
            self.cross_w = nn.ParameterList([nn.Parameter(torch.randn(din, din) * 0.01) for _ in range(3)])
            self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(din)) for _ in range(3)])
            self.deep = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.head = nn.Linear(din + 64, 1)

        def forward(self, u, m, g, a, o, gi, go):
            x0 = torch.cat([self.user_emb(u), self.movie_emb(m), self.gender_emb(g),
                            self.age_emb(a), self.occ_emb(o), self.genre_emb(gi, go)], dim=1)
            x = x0
            for w, b in zip(self.cross_w, self.cross_b, strict=True):
                x = x0 * (x @ w + b) + x  # DCN-v2 cross
            return self.head(torch.cat([x, self.deep(x0)], dim=1)).squeeze(1)

    model = DCN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    for _epoch in range(6):
        model.train()
        for u, m, g, a, o, gi, go, y in batchify(train):
            opt.zero_grad()
            lossf(model(u, m, g, a, o, gi, go), y).backward()
            opt.step()

    @torch.no_grad()
    def auc(mdl):
        mdl.eval()
        ps, ys = [], []
        for u, m, g, a, o, gi, go, y in batchify(val):
            ps.append(torch.sigmoid(mdl(u, m, g, a, o, gi, go)).cpu().numpy())
            ys.append(y.cpu().numpy())
        return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

    fp_auc = auc(model)
    print(f"\nfp AUC: {fp_auc:.4f}")

    # --- per-component int4 fake-quant (per-row symmetric) → AUC drop ---
    @torch.no_grad()
    def fq_(w, bits=4):
        qmax = 2 ** (bits - 1) - 1
        if w.dim() >= 2:
            s = w.abs().amax(dim=1, keepdim=True) / qmax + 1e-12
        else:
            s = w.abs().max() / qmax + 1e-12
        w.copy_((w / s).round().clamp(-qmax - 1, qmax) * s)

    components = {
        "emb_big (user+movie)": ["user_emb", "movie_emb"],
        "emb_side (g/age/occ/genre)": ["gender_emb", "age_emb", "occ_emb", "genre_emb"],
        "cross layers": ["cross_w", "cross_b"],
        "deep MLP": ["deep"],
        "head": ["head"],
    }

    def quantized(prefixes: list[str]) -> float:
        want = set(prefixes)
        q = copy.deepcopy(model)
        for name, p in q.named_parameters():
            if name.split(".")[0] in want:  # top-level module/param name
                fq_(p)
        return auc(q)

    results = {"fp": round(fp_auc, 4)}
    results["all-int4"] = round(quantized([p for ps in components.values() for p in ps]), 4)
    for label, prefixes in components.items():
        results[label] = round(quantized(prefixes), 4)

    print(f"\n{'=' * 64}\nPER-COMPONENT int4 FRAGILITY — DCN-v2 on MovieLens-1M\n{'=' * 64}")
    print(f"  {'component':30s}  AUC     ΔAUC vs fp")
    print(f"  {'fp (baseline)':30s}  {fp_auc:.4f}   —")
    for label in ["all-int4", *components]:
        drop = fp_auc - results[label]
        print(f"  {label:30s}  {results[label]:.4f}   {drop:+.4f}")
    return {"fp_auc": round(fp_auc, 4), "results": results}


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(run.remote(), indent=2, default=str))
