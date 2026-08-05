"""Candidate functional forms for L(N, D, b) (PLAN.md section 8).

Forms compete; selection is by leave-one-size-out extrapolation, never
in-sample fit (PLAN.md principle 7, select.py). All positive quantities are
parameterized in log-space internally for L-BFGS-B stability; exponents stay
linear with box bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Quantized bit widths present in the grid (bf16 = 16.0 is the reference arm).
DEFAULT_QUANT_BITS: tuple[float, ...] = (1.58, 2.0, 3.0, 4.0)
BF16_BITS = 16.0


def fmt_bits(b: float) -> str:
    """1.58 -> '1.58', 2.0 -> '2', matching Precision values."""
    return f"{b:g}"


@dataclass(frozen=True)
class ParamSpec:
    """One fit parameter. Natural-space bounds; `log=True` means the optimizer
    sees log(x) (positive quantities), else x directly (exponents)."""

    name: str
    lo: float
    hi: float
    log: bool
    init_lo: float
    init_hi: float

    def to_internal(self, x: float) -> float:
        return float(np.log(x)) if self.log else float(x)

    def to_natural(self, t: float) -> float:
        return float(np.exp(t)) if self.log else float(t)


#: cache of per-row width indices, keyed by (bits bytes, shape, quant widths);
#: the bits column is constant across the thousands of objective evals in a fit.
_IDX_CACHE: dict[tuple[bytes, tuple[int, ...], tuple[float, ...]], np.ndarray] = {}


def bits_index(bits: np.ndarray, quant_bits: tuple[float, ...]) -> np.ndarray:
    """Per-row index into (quant value vector + [bf16 slot]): row i gets
    j < len(quant_bits) for quantized width j, len(quant_bits) for bf16."""
    bits = np.asarray(bits, dtype=float)
    key = (bits.tobytes(), bits.shape, quant_bits)
    idx = _IDX_CACHE.get(key)
    if idx is None:
        flat = np.atleast_1d(bits).ravel()
        idx = np.full(flat.shape, -1, dtype=np.intp)
        idx[np.isclose(flat, BF16_BITS)] = len(quant_bits)
        for j, b in enumerate(quant_bits):
            idx[np.isclose(flat, b)] = j
        if np.any(idx < 0):
            unknown = sorted(set(np.round(flat[idx < 0], 4)))
            raise ValueError(f"bit widths {unknown} not covered by form (has {list(quant_bits)})")
        idx = idx.reshape(bits.shape)  # scalar in -> scalar-shaped out
        _IDX_CACHE[key] = idx
    return idx


class Form:
    """Base candidate form: param specs, internal bounds, deterministic LHS
    init sampler, predict(params, N, D, bits) -> L."""

    name: str = "base"

    def __init__(self, quant_bits: tuple[float, ...] = DEFAULT_QUANT_BITS):
        self.quant_bits = tuple(quant_bits)
        self.specs: list[ParamSpec] = self._build_specs()
        self._names = [s.name for s in self.specs]
        self._log_mask = np.array([s.log for s in self.specs])
        #: True when a subclass overrides penalty (fit.py skips it otherwise)
        self.has_penalty = type(self).penalty is not Form.penalty

    def _build_specs(self) -> list[ParamSpec]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- parameter plumbing ----

    @property
    def param_names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def n_params(self) -> int:
        return len(self.specs)

    def internal_bounds(self) -> list[tuple[float, float]]:
        return [(s.to_internal(s.lo), s.to_internal(s.hi)) for s in self.specs]

    def to_natural(self, theta: np.ndarray) -> dict[str, float]:
        nat = np.where(self._log_mask, np.exp(theta), theta)
        return dict(zip(self._names, nat.tolist()))

    def to_internal(self, params: dict[str, float]) -> np.ndarray:
        return np.array([s.to_internal(params[s.name]) for s in self.specs])

    def sample_inits(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Latin-hypercube in internal space over each spec's init range.
        Deterministic given rng state (fit.py seeds it)."""
        pts = np.empty((n, self.n_params))
        for j, s in enumerate(self.specs):
            lo, hi = s.to_internal(s.init_lo), s.to_internal(s.init_hi)
            u = (rng.permutation(n) + rng.uniform(size=n)) / n
            pts[:, j] = lo + u * (hi - lo)
        return pts

    # ---- model ----

    def predict(self, params: dict[str, float], N, D, bits) -> np.ndarray:
        raise NotImplementedError

    def predict_theta(self, theta: np.ndarray, N, D, bits) -> np.ndarray:
        return self.predict(self.to_natural(theta), N, D, bits)

    def penalty(self, params: dict[str, float]) -> float:
        """Extra objective term (Form C's L1); zero by default."""
        return 0.0

    def f_of_b(self, params: dict[str, float], bits) -> np.ndarray:
        """Effective-capacity multiplier f(b) if the form defines one."""
        raise ValueError(f"form {self.name!r} has no effective-capacity f(b)")


# ---- shared spec fragments (natural-space bounds / init ranges) ----

_E = ParamSpec("E", 1e-3, 5.0, True, 0.2, 2.0)
_A = ParamSpec("A", 1e-2, 1e6, True, 10.0, 1e4)
_ALPHA = ParamSpec("alpha", 0.05, 1.2, False, 0.1, 0.6)
_B = ParamSpec("B", 1e-2, 1e6, True, 10.0, 1e4)
_BETA = ParamSpec("beta", 0.05, 1.2, False, 0.1, 0.6)
_DELTA = ParamSpec("delta", 0.0, 1.0, False, 0.05, 0.5)
_ETA = ParamSpec("eta", 0.0, 1.0, False, 0.05, 0.5)
_B0 = ParamSpec("b0", -8.0, 1.5, False, -2.0, 1.2)
_GAMMA_F = ParamSpec("gamma_f", 0.05, 30.0, True, 0.5, 10.0)


def _f_spec(b: float) -> ParamSpec:
    return ParamSpec(f"f_{fmt_bits(b)}", 1e-3, 2.0, True, 0.2, 1.2)


def _g_spec(b: float) -> ParamSpec:
    return ParamSpec(f"g_{fmt_bits(b)}", 1e-10, 1e4, True, 1e-5, 1.0)


class FormAFree(Form):
    """Form A, effective capacity, f free per quantized width (PLAN.md s8):
    L = E + A / (N * f(b))^alpha + B / D^beta, with f(16) = 1 fixed."""

    name = "A-free"

    def _build_specs(self) -> list[ParamSpec]:
        return [_E, _A, _ALPHA, _B, _BETA] + [_f_spec(b) for b in self.quant_bits]

    def f_of_b(self, params: dict[str, float], bits) -> np.ndarray:
        vals = np.array([params[f"f_{fmt_bits(b)}"] for b in self.quant_bits] + [1.0])
        return vals[bits_index(np.asarray(bits, dtype=float), self.quant_bits)]

    def predict(self, params, N, D, bits) -> np.ndarray:
        N, D = np.asarray(N, dtype=float), np.asarray(D, dtype=float)
        n_eff = N * self.f_of_b(params, bits)
        return params["E"] + params["A"] / n_eff ** params["alpha"] + params["B"] / D ** params["beta"]


class FormAParam(Form):
    """Form A with parametric f(b) = 1 - exp(-(b - b0)/gamma_f) (Kumar-style),
    normalized by f(16) so the bf16 arm has f = 1 exactly."""

    name = "A-param"

    def _build_specs(self) -> list[ParamSpec]:
        return [_E, _A, _ALPHA, _B, _BETA, _B0, _GAMMA_F]

    def f_of_b(self, params: dict[str, float], bits) -> np.ndarray:
        bits = np.asarray(bits, dtype=float)
        b0, gam = params["b0"], params["gamma_f"]
        raw = 1.0 - np.exp(-(bits - b0) / gam)
        norm = 1.0 - np.exp(-(BF16_BITS - b0) / gam)
        return np.clip(raw / norm, 1e-9, None)

    def predict(self, params, N, D, bits) -> np.ndarray:
        N, D = np.asarray(N, dtype=float), np.asarray(D, dtype=float)
        n_eff = N * self.f_of_b(params, bits)
        return params["E"] + params["A"] / n_eff ** params["alpha"] + params["B"] / D ** params["beta"]


class FormB(Form):
    """Form B, additive degradation (PLAN.md s8):
    L = E + A/N^alpha + B/D^beta + g(b) * D^delta / N^eta, g free per
    quantized width, g(16) = 0. The overall coefficient C is absorbed into
    g(b) (C and g only appear as a product, so C is not identifiable);
    Ouyang-style: the gap grows with tokens, shrinks with size."""

    name = "B"

    def _build_specs(self) -> list[ParamSpec]:
        return [_E, _A, _ALPHA, _B, _BETA, _DELTA, _ETA] + [_g_spec(b) for b in self.quant_bits]

    def g_of_b(self, params: dict[str, float], bits) -> np.ndarray:
        vals = np.array([params[f"g_{fmt_bits(b)}"] for b in self.quant_bits] + [0.0])
        return vals[bits_index(np.asarray(bits, dtype=float), self.quant_bits)]

    def predict(self, params, N, D, bits) -> np.ndarray:
        N, D = np.asarray(N, dtype=float), np.asarray(D, dtype=float)
        chinchilla = params["E"] + params["A"] / N ** params["alpha"] + params["B"] / D ** params["beta"]
        return chinchilla + self.g_of_b(params, bits) * D ** params["delta"] / N ** params["eta"]


class FormC(Form):
    """Form C, penalized combination (PLAN.md s8): both the effective-capacity
    term (parametric f, keeps the form identifiable) and the additive
    degradation term, with an L1 penalty lam * sum_b g(b) on the degradation
    coefficients so the data must earn the extra term. With g -> 0 this
    collapses to Form A-param; the penalty weight `lam` trades bias for
    parsimony and is reported with any Form C fit."""

    name = "C"

    def __init__(self, quant_bits: tuple[float, ...] = DEFAULT_QUANT_BITS, lam: float = 1e-3):
        self.lam = lam
        super().__init__(quant_bits)

    def _build_specs(self) -> list[ParamSpec]:
        return [_E, _A, _ALPHA, _B, _BETA, _B0, _GAMMA_F, _DELTA, _ETA] + [
            _g_spec(b) for b in self.quant_bits
        ]

    f_of_b = FormAParam.f_of_b
    g_of_b = FormB.g_of_b

    def predict(self, params, N, D, bits) -> np.ndarray:
        N, D = np.asarray(N, dtype=float), np.asarray(D, dtype=float)
        n_eff = N * self.f_of_b(params, bits)
        cap = params["E"] + params["A"] / n_eff ** params["alpha"] + params["B"] / D ** params["beta"]
        return cap + self.g_of_b(params, bits) * D ** params["delta"] / N ** params["eta"]

    def penalty(self, params: dict[str, float]) -> float:
        return self.lam * sum(abs(params[f"g_{fmt_bits(b)}"]) for b in self.quant_bits)


def default_forms(quant_bits: tuple[float, ...] = DEFAULT_QUANT_BITS) -> dict[str, Form]:
    """The competing families (PLAN.md principle 7: at least two)."""
    return {
        f.name: f
        for f in (FormAFree(quant_bits), FormAParam(quant_bits), FormB(quant_bits), FormC(quant_bits))
    }
