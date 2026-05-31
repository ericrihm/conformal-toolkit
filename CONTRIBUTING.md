# Contributing

Thanks for your interest — corrections and contributions are genuinely welcome.
This project values being *correctable*: if you find an error in the math, the
code, the docs, or in [`ERRATA.md`](ERRATA.md) itself, that's the system working.

## Reporting a math/correctness error (errata)

This is the highest-value contribution. Follow the same method the project's own
audit used (see [`ERRATA.md` → "How we caught them"](ERRATA.md)):

1. Open an issue titled `Errata: <one-line claim>`.
2. Give the counter-evidence as concretely as possible:
   - a **specific geometry** (a sphere radius, a flat patch, an explicit metric)
     on which the claim returns the wrong number; or
   - a **conformal-weight argument** showing the terms can't match.
   A failing check on a named anchor metric is the gold standard.
3. Propose the corrected formula, **stating your normalization convention**
   (Branson vs. analyst signs differ — many conformal-geometry "errors" are
   convention clashes), with a reference if you have one.
4. If you can, add a regression test pinning the right value on the anchor —
   *verified-on-an-anchor beats argued-in-prose.*

Open mathematical problems where help is explicitly wanted are listed at the end
of [`ERRATA.md`](ERRATA.md) (e.g. the complete weight-4 hypersurface invariant
basis, the Fialkow/Weyl terms in `L₂`, the full extrinsic `Q₄`).

## Development & tests

The suite runs on two independent tracks (see the README section
*"How it's tested"*):

```bash
# Track B — discrete (PyTorch only, no SageMath needed):
pip install -e ".[dev,ml]"
pytest tests/test_discrete/ tests/test_features/ -v

# Track A — symbolic (needs SageMath 10.x):
./sage-run.sh test          # auto-detects native micromamba or the Sage Docker image
# or target specific modules:
./sage-run.sh pytest tests/test_core/ -v
```

Both tracks run automatically on every push and pull request via GitHub Actions
([`.github/workflows/test.yml`](.github/workflows/test.yml)) — please make sure
they're green. If you add or correct a formula, add a regression anchor (a known
closed-form value on a concrete geometry) alongside it; see
`tests/test_discrete/test_errata_anchors.py` for the pattern.

## Conventions

- Match the surrounding code style; keep changes focused.
- Tag any in-code note about a known limitation or correction with a stable
  `# ERRATA <ID>` reference so it stays traceable to the errata log.
- By contributing you agree your contributions are licensed under the MIT License.
