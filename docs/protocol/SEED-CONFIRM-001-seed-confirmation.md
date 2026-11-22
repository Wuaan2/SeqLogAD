# SEED-CONFIRM-001 — Confirmation seeds for Phase 2

Preregistered 2026-11-22, before any confirmation run. Machine-readable: `configs/protocols/seed-confirmation-v1.yaml`.

## Purpose

Phase 2 rests on one training seed (limitation L1 of P2-FREEZE-001). This protocol retrains the four experts
with seeds **3407** and **8675309** to report how stable the Phase-2 findings are. **Seed 42 stays the primary
seed**: every decision, every frozen artifact and every Phase-3 input comes from seed 42. Nothing here changes
P2-FREEZE-001, its dispositions, its selected checkpoints or the G2 decision.

## What is run

| Task | Confirmation runs |
|---|---|
| P2.1 Semantic | Retrained under its frozen final protocol, which already lists seeds 42, 3407 and 8675309 |
| P2.2 Sequence reference | Retrained; V13R1 config with only the training seed changed |
| P2.3 Lightweight | Retrained; LIGHT-SEQ-001 config with only the training seed changed |
| P2.4 GTAT | Retrained; GTAT-ST-001 config with only the training seed changed |
| P2.5 Evidence contract | Not rerun; EVID-CS-001 is unchanged |
| P2.6 Diversity | Each seed's selected checkpoints score the same held-out windows; the DIV-P2-001 analysis is rerun unchanged |

## Held fixed across seeds

- Data bundles, label sidecar and graph view (the P2-FREEZE-001 artifacts).
- Expert configs, step budget, early stopping and precision.
- For P2.2–P2.4, the checkpoint-selection windows and rule: the selection windows are those of seed 42 for every
  training seed, so the P2.6 held-out windows (validation minus selection windows) remain unseen by every
  checkpoint selection.
- P2.6 held-out windows, comparators, metrics, alarm budget, bootstrap (2000 replicates, seed 42) and verdict rule.

What varies with the seed: weight initialisation, dropout, data order, training-window sampling, and for P2.1 the
masking and its label-free validation records (as its own frozen protocol defines).

## Code

New package `src/seqlogad/seeds`; the seed-42 modules are reused unchanged.

- `runner.py` trains one expert for both folds and one seed, with the same Drive persistence and recovery as seed 42.
- `adapters.py` runs the frozen P2.2–P2.4 trainers with the confirmation seed, the seed-42 selection windows and
  seed-named run ids, without editing them.
- `score.py` runs the frozen P2.6 scorer on the seed's FINAL ZIPs.

## Reporting

Per seed: AP, AUROC, FAR@R0.5, FAR@R0.8, selected checkpoint and every DIV-P2-001 verdict. Across seeds: mean,
minimum and maximum. A finding is called **seed-stable** only if it holds in all three seeds. A run that fails is
reported as missing and is never replaced by another seed. The report (`docs/reports/SEED-CONFIRM-001.md`) is
written only after all runs finish.

## Forbidden

- Changing P2-FREEZE-001 dispositions, selected checkpoints or the G2 decision because of these runs.
- Choosing a best seed or averaging weights across seeds.
- Feeding confirmation-seed results into any Phase-3 choice.
- Reading target labels.
