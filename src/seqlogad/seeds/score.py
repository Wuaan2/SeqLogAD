"""P2.6 scoring pass for a confirmation seed (SEED-CONFIRM-001).

Reuses the frozen scorer `seqlogad.diversity.score` unchanged (it is bound by P2-FREEZE-001). Only three
names are swapped for the call: the expected FINAL ZIP digests (taken from the seed's own FINAL ZIPs), the
seed in the target-ZIP path, and, for P2.4, the training seed in the config identity. Same held-out windows
(selection seed 42), same EVID-CS-001 envelopes, same sanity-reproduction guard.
"""
from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from seqlogad.common.checksum import sha256_file

from .protocol import CONFIRMATION_SEEDS


def seeded_extract_run(seed, digests):
    def extract_run(final_zip, task, target, work_root):
        if sha256_file(final_zip) != digests[task]:
            raise ValueError(f"{task} S{seed} FINAL ZIP changed since the scoring pass started")
        destination = Path(work_root) / task / target
        if not (destination / "EXTRACTED").is_file():
            shutil.rmtree(destination, ignore_errors=True)
            destination.mkdir(parents=True)
            with zipfile.ZipFile(final_zip) as outer:
                payload = outer.read(f"{task}_S{seed}/targets/{task}_{target}_S{seed}.zip")
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                for name in inner.namelist():
                    member = PurePosixPath(name)
                    if member.is_absolute() or ".." in member.parts or "\\" in name:
                        raise ValueError("unsafe member in target ZIP")
                inner.extractall(destination)
            (destination / "EXTRACTED").write_text("ok\n")
        runs = [path for path in destination.iterdir() if path.is_dir()]
        if len(runs) != 1 or not runs[0].name.endswith((f"-S{seed}", f"-S{seed}-V13R1")):
            raise ValueError(f"target ZIP must hold exactly one S{seed} run directory")
        return runs[0]
    return extract_run


@contextmanager
def seeded_scoring(seed, finals):
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("not a preregistered confirmation seed")
    from seqlogad.diversity import score as frozen
    from seqlogad.graph import config as gconfig

    digests = {task: sha256_file(path) for task, path in finals.items()}
    load_config = gconfig.load_config
    originals = (frozen.FINAL_SHA256, frozen.extract_run)
    frozen.FINAL_SHA256, frozen.extract_run = digests, seeded_extract_run(seed, digests)
    gconfig.load_config = lambda repo_root: dict(load_config(repo_root), seed=seed)
    try:
        yield frozen
    finally:
        frozen.FINAL_SHA256, frozen.extract_run = originals
        gconfig.load_config = load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=CONFIRMATION_SEEDS)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--semantic-root", help="semantic bundle root; needed by P2.1")
    parser.add_argument("--graph-root", help="graph view root; needed by P2.4")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--final", action="append", required=True, help="TASK=path to the seed's FINAL ZIP")
    parser.add_argument("--tasks", default="P2.4,P2.3,P2.2,P2.1")
    parser.add_argument("--targets", default="BGL,HDFS")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    finals = dict(item.split("=", 1) for item in args.final)
    tasks = args.tasks.split(",")
    with seeded_scoring(args.seed, {task: finals[task] for task in tasks}) as frozen:
        for task in tasks:
            for target in args.targets.split(","):
                frozen.score_task(task, target, repo_root=args.repo_root, data_root=args.data_root,
                                  final_zip=finals[task], output_root=args.output_root, work_root=args.work_root,
                                  semantic_root=args.semantic_root, graph_root=args.graph_root, device=args.device)
    print(f"P2.6_SCORING_S{args.seed}=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
