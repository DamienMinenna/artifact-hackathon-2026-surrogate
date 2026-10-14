"""Log an official score of the surrogate-models challenge to W&B, for any team.

The scores are the organisers' ones, from `pallas_score.py` run on the hidden test. This
script only transcribes them into the team's group, in the shape the leaderboard reads
(`pallas/mean_r2` in the run summary, `task` and `tier` in its config). It computes nothing.

    python log_official_score.py --team plasma-pirates --task ood --tier hard \\
        --model grl_trajectory_torch_mlp_ood --model-version 1 \\
        --submission submissions/grl_trajectory_torch_mlp_ood_v1/submission_ood.csv \\
        --score mean=0.9513 --score E_MeV=0.9829 --score in_distribution.mean=0.9801 --dry-run

Scores come from `--scores FILE`, `--score NAME=VALUE` (repeatable), or both:
  - FILE is JSON, `{"E_MeV": 0.99, "mean": 0.98, "n_rows": 14573}` as `score_submission`
    returns it, optionally with sections: `{"mean": ..., "drift": {...}, "in_distribution":
    {...}, "drop": {...}}`; or the text `pallas_score.py` prints (`name  value` per line).
  - NAME maps as the existing official runs do: `mean` -> `pallas/mean_r2`, a target ->
    `pallas/r2/<target>`, `n_*` and `median_*` -> `pallas/<name>`; `section.name` puts it under
    `pallas/<section>/` (`drop` keeps plain names); a name starting with `pallas/` is kept.

The key is WANDB_API_KEY, or the first `.env` found in this folder or above it. Use
`--dry-run` first: it prints the run without logging it. A run of the same name in the
team's group is refused unless `--allow-duplicate`.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

TIERS = ("easy", "medium", "hard", "optional")
RANKED_SECTIONS = ("", "in_plasma")      # the ranked score sits at the top level


def load_env():
    """Set the variables of the nearest `.env` that are not already in the environment."""
    for folder in [Path.cwd(), *Path.cwd().parents]:
        path = folder / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                name, sep, value = line.strip().partition("=")
                if sep and name and not name.startswith("#"):
                    os.environ.setdefault(name.strip(), value.strip().strip("'\""))
            return path
    return None


def wandb_key(name, section=""):
    """The W&B key of one scorer value, as the existing official runs name it."""
    if name.startswith("pallas/"):
        return name
    if "." in name and not section:
        section, name = name.split(".", 1)
    prefix = "pallas/" if section in RANKED_SECTIONS else f"pallas/{section}/"
    if section == "drop" or name.startswith(("n_", "median_")):
        return prefix + name
    return prefix + ("mean_r2" if name == "mean" else f"r2/{name}")


def read_scores(path):
    """{W&B key: value} from a JSON file or from the text pallas_score.py prints."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and not parts[0].endswith(":"):
                try:
                    data[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    scores = {}
    for name, value in data.items():
        if isinstance(value, dict):
            for inner, number in value.items():
                scores[wandb_key(inner, name)] = number
        else:
            scores[wandb_key(name)] = value
    return scores


def parse_pairs(pairs, what):
    out = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            sys.exit(f"{what} must be NAME=VALUE, got {pair!r}")
        out[name.strip()] = value.strip()
    return out


def number(value):
    value = float(value)
    return int(value) if value.is_integer() and abs(value) > 1 else value


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 2)[2])
    parser.add_argument("--team", default=os.environ.get("WANDB_GROUP"),
                        help="the team slug, i.e. the W&B group (default: $WANDB_GROUP)")
    parser.add_argument("--task", required=True,
                        help="trajectory, direct_B, inverse_A, inverse_B, moments, ood, ...")
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument("--scores", type=Path, help="JSON or pallas_score.py text output")
    parser.add_argument("--score", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--model", help="the model that made the submission")
    parser.add_argument("--model-version", type=int)
    parser.add_argument("--submission", type=Path,
                        help="the CSV that was scored (its name and md5 go in the config)")
    parser.add_argument("--scored-at", default=datetime.now().strftime("%Y-%m-%d %H:%M"))
    parser.add_argument("--scored-on", default="hidden test (organisers)")
    parser.add_argument("--section", help="free text: which part of objectives.md")
    parser.add_argument("--config", action="append", default=[], metavar="NAME=VALUE",
                        help="extra config fields, e.g. registry_project=m4cast_plasma_pirates")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--name", help="run name (default official-<task>-<model>-v<version>)")
    parser.add_argument("--entity", default="artifact-hackathon-2026")
    parser.add_argument("--project", default="surrogate-models")
    parser.add_argument("--dry-run", action="store_true", help="print the run, don't log it")
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    if not args.team:
        sys.exit("--team is required (your team slug, exactly as on your Agora-Conf page)")
    scores = read_scores(args.scores) if args.scores else {}
    scores.update({wandb_key(k): v for k, v in parse_pairs(args.score, "--score").items()})
    scores = {k: number(v) for k, v in scores.items()}
    if not any(k == "pallas/mean_r2" or k.startswith("pallas/median_") for k in scores):
        sys.exit("no ranked score: give the mean (--score mean=0.95) or a median_* key")

    config = {"task": args.task, "tier": args.tier, "scored_at": args.scored_at,
              "scored_on": args.scored_on}
    if args.section:
        config["score_section"] = args.section
    if args.model:
        config["model_name"] = args.model
    if args.model_version is not None:
        config["model_version"] = args.model_version
    if args.submission:
        if not args.submission.is_file():
            sys.exit(f"{args.submission} not found")
        config["submission_file"] = args.submission.name
        config["submission_md5"] = hashlib.md5(args.submission.read_bytes()).hexdigest()
    config.update(parse_pairs(args.config, "--config"))
    label = args.model or (args.submission.stem if args.submission else "submission")
    version = f"-v{args.model_version}" if args.model_version is not None else ""
    name = args.name or f"official-{args.task}-{label}{version}"
    tags = sorted({"official", args.task, *args.tag})

    print(f"run     {name}\nwhere   {args.entity}/{args.project}, group {args.team}\n"
          f"tags    {', '.join(tags)}\nconfig  {json.dumps(config, indent=2)}\nscores")
    for key, value in sorted(scores.items()):
        print(f"  {key:40s} {value}")

    load_env()
    if not os.environ.get("WANDB_API_KEY"):
        sys.exit("WANDB_API_KEY is not set and no .env with it was found")
    import wandb
    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=60)
    same = [r for r in api.runs(f"{args.entity}/{args.project}", filters={"group": args.team})
            if r.name == name]
    if same and not args.allow_duplicate:
        sys.exit(f"\n{args.team} already has a run named {name}: {same[0].url}\n"
                 "change --name, or pass --allow-duplicate to log another one")
    if args.dry_run:
        print("\ndry run: nothing logged")
        return

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(entity=args.entity, project=args.project, group=args.team, name=name,
                     tags=tags, config=config, job_type="official-score")
    run.log(scores)
    run.summary.update(scores)
    run.finish()
    print(f"\nlogged: {run.url}")


if __name__ == "__main__":
    main()
