# Contributing to spinoct

Thanks for your interest. `spinoct` is the optimal-control engine behind the Espira research product.

## Ground rules

- **The positive controls are sacred.** Every numerical solver is validated against the closed-form
  analytic results in `spinoct.analytic` before it is trusted. A change that makes a numerical result
  disagree with an analytic identity is a bug in the change, not in the identity, until proven
  otherwise.
- **Units are declared, never assumed.** New physical constants live in `spinoct.units` with a source
  and a unit. `scripts/check_unit_constants.py` fails the build on any unreviewed dimensional
  constant elsewhere.
- **The switching cost is not an energy.** It is tesla-squared-seconds. It becomes joules only through
  an explicit `CircuitModel`. Do not add a hidden conversion.

## Development

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,torch]"
.venv/Scripts/python -m ruff check src tests scripts
.venv/Scripts/python scripts/check_unit_constants.py
.venv/Scripts/python -m pytest tests -q
```

## Pull requests

Small, focused changes with tests. English only in code, comments and docs. Reference the primary
source (a DOI) for any new physics.
