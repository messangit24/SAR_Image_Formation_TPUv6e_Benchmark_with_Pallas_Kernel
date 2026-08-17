---
name: write-python
description: Authoritative Python coding standards, typing rules, lazy import patterns, and test practices for developers and AI agents. Enforced by dev-story, /owl, and review agents.
---

# Python Engineering & Quality Standards (`write-python`)

Use this skill when authoring, refactoring, or reviewing Python code in this repository.

---

## 🏛️ 1. Type Annotations & Modern Idioms (Python 3.12+)

- **Modern Built-in Generics (PEP 585 & 604)**:
  - Use `list[str]`, `dict[str, Any]`, `set[int]`, `tuple[int, ...]` instead of legacy `typing.List`, `typing.Dict`.
  - Use `str | None` or `int | float` instead of `typing.Optional` / `typing.Union`.
- **Strict Public Signatures**: Every public function, method, and module export MUST have explicit argument and return type annotations.
- **Pydantic v2 & Dataclasses**: Prefer Pydantic `BaseModel` (v2) or `@dataclass(frozen=True, slots=True)` for structured data transfer objects.
- **No Untyped Any Leaks**: If `Any` is required at an external boundary, validate and cast it to a concrete type immediately.

---

## ⚡ 2. Performance & Startup: Lazy Imports

Heavy machine learning, scientific, or serialization libraries (e.g. `torch`, `jax`, `tensorflow`, `cv2`, `pandas`, `polars`, `transformers`) **MUST be imported lazily inside the functions that use them**:

```python
# ❌ Anti-pattern: Top-level heavy import slows down CLI startup and pytest
import torch
import transformers

def run_inference(data: list[float]) -> dict[str, Any]:
    ...

# ✅ Recommended: Lazy import inside execution scope
def run_inference(data: list[float]) -> dict[str, Any]:
    import torch
    from transformers import AutoModel
    ...
```

*Rationale*: Keeps test discovery, linting, and CLI startup under $100\text{ms}$.

---

## 📝 3. Documentation & "Why" Comments

- **Module Docstrings**: Every `.py` file must begin with a concise docstring describing its role, inputs, and outputs.
- **Explain "Why", Not "What"**: Comments must explain the rationale behind non-obvious algorithms, trade-offs, bug workarounds, or hardware constraints.
- **Preserve Existing Documentation**: Never strip existing comments or docstrings during refactoring unless explicitly obsolete.

---

## 🛡️ 4. Security & Robustness Invariants

- **Safe Path Operations**: Always use `pathlib.Path` and `.resolve()`. Validate against path traversal attacks.
- **Hermetic Serialization**: Use `json.loads` or `yaml.safe_load`. Never use `pickle.loads` or raw `eval()`.
- **Zero Raw SQL Formatting**: Use parameterized queries (`text("SELECT ... WHERE id = :id")`) or ORMs.
- **Explicit Error Handling**: Define custom domain exceptions inheriting from a base project error. Never catch bare `except:` or silently swallow exceptions with `pass`.

---

## 🧪 5. Test-Driven Development & Hermeticity

- **Test Placement & Naming**: Tests reside under `tests/test_<module_name>.py`.
- **Zero Network in Unit Tests**: All external HTTP calls, GCP services, and DB connections must use hermetic mocks, monkeypatching, or local in-memory fixtures.
- **Deterministic Assertions**: Use explicit equality assertions with helpful error messages.
