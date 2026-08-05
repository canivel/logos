"""lm-eval-harness wrapper, version pinned (PLAN.md s4: "lm-eval-harness,
version pinned in the repo"). Benchmarks are the validator, loss is the
fitting target (principle 3).

`lm_eval` is an optional dependency: it is imported lazily and a clear
install error is raised if missing. The version pin is enforced hard —
results from a different harness version are not comparable and the
validation panel checks the recorded pin.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

LM_EVAL_PIN = "0.4.9"

# Task suites (PLAN.md s6, s7, s9, s11).
SMALL_SUITE = ("arc_easy", "hellaswag", "piqa", "sciq", "lambada_openai")  # P1, 0-shot
FULL_SUITE = SMALL_SUITE + ("arc_challenge", "winogrande", "boolq", "openbookqa")  # P2+
P3_SUITE = FULL_SUITE + ("mmlu",)  # mmlu runs 5-shot: pass num_fewshot=5 for it
CAPSTONE_SUITE = P3_SUITE + ("gsm8k", "ifeval")  # P5, post-SFT
FEWSHOT = {"mmlu": 5}  # PLAN.md s9: MMLU 5-shot at 1B+

# Preference order when extracting one headline metric per task.
_METRIC_KEYS = ("acc_norm,none", "acc,none", "exact_match,none", "perplexity,none")


def _require_lm_eval():
    """Import lm_eval or fail loudly; refuse any version but the pin."""
    try:
        import lm_eval
    except ImportError as e:
        raise ImportError(
            "lm-eval-harness is required for downstream evals but is not installed. "
            f"Install the pinned version: `uv pip install lm-eval=={LM_EVAL_PIN}`. "
            "The pin is a design rule (PLAN.md s4): results from other harness "
            "versions are not comparable across the TRIT suite."
        ) from e
    found = getattr(lm_eval, "__version__", None)
    if found is None:
        from importlib.metadata import version

        found = version("lm_eval")
    if found != LM_EVAL_PIN:
        raise RuntimeError(
            f"lm-eval version mismatch: found {found}, pinned {LM_EVAL_PIN}. "
            f"Refusing to run. `uv pip install lm-eval=={LM_EVAL_PIN}`."
        )
    return lm_eval


def build_harness_lm(
    model,
    tokenizer,
    device: str = "cpu",
    batch_size: int = 8,
    max_seq_len: int | None = None,
    max_gen_toks: int = 256,
):
    """Wrap a logos Transformer + HF tokenizer as an lm_eval.api.model.LM.

    The class is defined lazily so this module imports cleanly without
    lm_eval installed. Generation is greedy and simple by design — the
    suites here are loglikelihood-dominated below the capstone.
    """
    _require_lm_eval()
    from lm_eval.api.model import LM

    class LogosLM(LM):
        def __init__(self):
            super().__init__()
            self.model = model.to(device).eval()
            self.tok = tokenizer
            self.dev = device
            self.max_len = max_seq_len or getattr(model.cfg, "max_seq_len", 2048)
            self.bs = batch_size
            eot = getattr(tokenizer, "eos_token_id", None)
            self.eot = int(eot) if eot is not None else 0

        # ---- helpers ----

        def _encode(self, s: str) -> list[int]:
            return self.tok.encode(s, add_special_tokens=False)

        @torch.no_grad()
        def _logits(self, ids: list[int]) -> torch.Tensor:
            x = torch.tensor([ids], dtype=torch.long, device=self.dev)
            out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            return logits[0].float()  # [T, V]

        @torch.no_grad()
        def _score(self, ctx_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
            """Sum logprob of cont_ids given ctx_ids, plus greedy-match flag."""
            ids = (ctx_ids + cont_ids)[-(self.max_len + 1) :]
            if len(ids) <= len(cont_ids):  # context fully truncated away
                ids = [self.eot] + ids[-(self.max_len) :]
            logits = self._logits(ids[:-1])
            lp = F.log_softmax(logits, dim=-1)
            n = len(cont_ids)
            tgt = torch.tensor(ids[-n:], dtype=torch.long, device=self.dev)
            rows = lp[-n:]
            ll = float(rows.gather(-1, tgt[:, None]).sum().item())
            greedy = bool((rows.argmax(-1) == tgt).all().item())
            return ll, greedy

        # ---- LM interface (lm_eval 0.4.x) ----

        def loglikelihood(self, requests) -> list[tuple[float, bool]]:
            out = []
            for req in requests:
                context, continuation = req.args
                whole = self._encode(context + continuation)
                ctx = self._encode(context) if context else [self.eot]
                if context and whole[: len(ctx)] == ctx:
                    cont = whole[len(ctx) :]
                else:
                    cont = self._encode(continuation)
                if not cont:
                    cont = whole[-1:] or [self.eot]
                    ctx = whole[:-1] or [self.eot]
                out.append(self._score(ctx, cont))
            return out

        def loglikelihood_rolling(self, requests) -> list[float]:
            """Full-document NLL over non-overlapping max_len windows; an EOT
            prefix lets the first token be scored (harness convention)."""
            out = []
            for req in requests:
                (text,) = req.args
                toks = self._encode(text)
                ids = [self.eot] + toks
                total = 0.0
                for start in range(0, len(toks), self.max_len):
                    cont = toks[start : start + self.max_len]
                    ctx = [ids[start]]  # window's single preceding token
                    total += self._score(ctx, cont)[0]
                out.append(total)
            return out

        @torch.no_grad()
        def generate_until(self, requests) -> list[str]:
            out = []
            for req in requests:
                context, kwargs = req.args
                until = list(kwargs.get("until", []) or [])
                max_gen = int(kwargs.get("max_gen_toks", max_gen_toks))
                ids = self._encode(context)[-(self.max_len - max_gen) :]
                gen: list[int] = []
                for _ in range(max_gen):
                    logits = self._logits(ids + gen)
                    nxt = int(logits[-1].argmax().item())
                    if nxt == self.eot:
                        break
                    gen.append(nxt)
                    text = self.tok.decode(gen)
                    if any(s in text for s in until):
                        break
                text = self.tok.decode(gen)
                for s in until:
                    if s in text:
                        text = text.split(s)[0]
                out.append(text)
            return out

    return LogosLM()


def run_downstream(
    model,
    tokenizer,
    tasks,
    device: str = "cpu",
    limit: int | None = None,
    batch_size: int = 8,
    num_fewshot: int | None = None,
    out_path: str | Path | None = None,
) -> dict[str, float]:
    """Run lm-eval tasks, return {task: headline metric}.

    Records the lm-eval version and per-task versions in the results json
    (the validation panel cross-checks the pin). For P3_SUITE mmlu, call
    with num_fewshot=FEWSHOT['mmlu'] on its own pass.
    """
    lm_eval = _require_lm_eval()
    lm = build_harness_lm(model, tokenizer, device=device, batch_size=batch_size)
    res = lm_eval.simple_evaluate(
        model=lm, tasks=list(tasks), limit=limit, num_fewshot=num_fewshot
    )
    metrics: dict[str, float] = {}
    for task, r in res["results"].items():
        for key in _METRIC_KEYS:
            if key in r and isinstance(r[key], (int, float)):
                metrics[task] = float(r[key])
                break
        else:
            for key, v in r.items():
                if isinstance(v, (int, float)) and not key.endswith("_stderr"):
                    metrics[task] = float(v)
                    break
    if out_path is not None:
        payload = {
            "lm_eval_version": LM_EVAL_PIN,
            "task_versions": res.get("versions", {}),
            "tasks": list(tasks),
            "limit": limit,
            "num_fewshot": num_fewshot,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": metrics,
            "results": res["results"],
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return metrics
