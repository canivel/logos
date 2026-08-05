"""`logos` command line.

SUBCOMMANDS is a module-level registry so later modules (fit, eval, manifest,
validate) register themselves without touching this file:

    from logos import cli
    cli.register("fit", configure_fn, run_fn)

configure_fn(parser) adds arguments; run_fn(args) -> int exit code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import yaml

from logos.config import RunSpec

Configure = Callable[[argparse.ArgumentParser], None]
Run = Callable[[argparse.Namespace], int]

SUBCOMMANDS: dict[str, tuple[Configure, Run]] = {}


def register(name: str, configure: Configure, run: Run) -> None:
    SUBCOMMANDS[name] = (configure, run)


# ---------------------------------------------------------------------------
# data prepare
# ---------------------------------------------------------------------------


def _configure_data(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="data_cmd", required=True)
    prep = sub.add_parser("prepare", help="FineWeb-Edu -> tokenized shards")
    prep.add_argument("--out", required=True, help="output shard directory")
    prep.add_argument("--synthetic", action="store_true", help="deterministic pseudo-text, no network")
    prep.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    prep.add_argument("--dataset-config", default="sample-350BT")
    prep.add_argument("--tokenizer", default="mistralai/Mistral-7B-v0.1")
    prep.add_argument("--shard-tokens", type=int, default=100_000_000)
    prep.add_argument("--target-tokens", type=int, default=None)
    prep.add_argument("--val-docs", type=int, default=20_000)
    prep.add_argument("--seed", type=int, default=1337)
    # synthetic-mode knobs
    prep.add_argument("--n-shards", type=int, default=2)
    prep.add_argument("--vocab-size", type=int, default=32_768)
    prep.add_argument("--val-tokens", type=int, default=16_384)


def _run_data(args: argparse.Namespace) -> int:
    from logos.data.prepare import prepare_fineweb, prepare_synthetic

    if args.data_cmd == "prepare":
        if args.synthetic:
            index = prepare_synthetic(
                args.out,
                n_shards=args.n_shards,
                shard_tokens=args.shard_tokens,
                vocab_size=args.vocab_size,
                seed=args.seed,
                val_tokens=args.val_tokens,
            )
        else:
            index = prepare_fineweb(
                args.out,
                dataset=args.dataset,
                dataset_config=args.dataset_config,
                tokenizer_name=args.tokenizer,
                shard_tokens=args.shard_tokens,
                target_tokens=args.target_tokens,
                val_docs=args.val_docs,
                seed=args.seed,
            )
        print(json.dumps({k: index[k] for k in ("dataset", "tokenizer", "total_tokens")}))
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _configure_train(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", help="versioned run-manifest YAML (list of RunSpec rows)")
    p.add_argument("--run-id", help="row to launch (manifest mode) or name (inline mode)")
    # inline mode: one RunSpec built from flags
    p.add_argument("--size", choices=None, help="ladder key, e.g. 25m")
    p.add_argument("--precision", help="bf16 | 4 | 3 | 2 | 1.58")
    p.add_argument("--tokens-per-param", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="p0")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--lr-rules", help="frozen P1 LR rules YAML (precision -> multiplier)")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-tokens", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=10)


def _run_train(args: argparse.Namespace) -> int:
    from logos.train.trainer import train

    if args.manifest:
        if not args.run_id:
            raise SystemExit("--manifest requires --run-id")
        doc = yaml.safe_load(Path(args.manifest).read_text())
        rows = doc["runs"] if isinstance(doc, dict) else doc
        try:
            row = next(r for r in rows if r["run_id"] == args.run_id)
        except StopIteration:
            raise SystemExit(f"run_id {args.run_id!r} not in {args.manifest}") from None
        spec = RunSpec(**row)
    else:
        if not (args.size and args.precision and args.tokens_per_param is not None):
            raise SystemExit("need --manifest/--run-id or --size --precision --tokens-per-param")
        run_id = args.run_id or (
            f"{args.size}-{args.precision}-tp{args.tokens_per_param:g}-s{args.seed}"
        )
        spec = RunSpec(
            run_id=run_id,
            phase=args.phase,
            size=args.size,
            precision=args.precision,
            tokens_per_param=args.tokens_per_param,
            seed=args.seed,
        )
    lr_rules = yaml.safe_load(Path(args.lr_rules).read_text()) if args.lr_rules else None
    status = train(
        spec,
        data_dir=args.data_dir,
        run_dir=Path(args.runs_dir) / spec.run_id,
        lr_rules=lr_rules,
        device=args.device,
        batch_tokens_override=args.batch_tokens,
        log_interval=args.log_interval,
    )
    print(json.dumps(status))
    return 0


# ---------------------------------------------------------------------------
# manifest generate / validate (thin delegations)
# ---------------------------------------------------------------------------


def _configure_manifest(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="manifest_cmd", required=True)
    gen = sub.add_parser("generate", help="write P0-P5 manifests + summary.md")
    gen.add_argument("--out", default="manifests")


def _run_manifest(args: argparse.Namespace) -> int:
    from logos.manifest.generate import main as gen_main

    return gen_main(["--out", args.out]) or 0


def _configure_validate(p: argparse.ArgumentParser) -> None:
    p.add_argument("--all", action="store_true")
    p.add_argument("--probe", nargs="*", default=None)
    p.add_argument("--list", action="store_true")


def _run_validate(args: argparse.Namespace) -> int:
    from validation.panel import main as validate_main

    argv = (
        ["--list"] if args.list else (["--all"] if args.all else ["--probe", *(args.probe or [])])
    )
    return validate_main(argv)


register("data", _configure_data, _run_data)
register("train", _configure_train, _run_train)
register("manifest", _configure_manifest, _run_manifest)
register("validate", _configure_validate, _run_validate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="logos", description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    for name, (configure, run) in SUBCOMMANDS.items():
        sp = subs.add_parser(name)
        configure(sp)
        sp.set_defaults(_run=run)
    args = parser.parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    raise SystemExit(main())
