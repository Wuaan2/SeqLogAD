# Primary context

## 1. What is authoritative?

`Bang_ke_hoach_SeqLogAD.xlsx` — the researcher's workbook — is the primary and
authoritative scientific roadmap. Where it conflicts with any other document, the
workbook wins.

Machine-readable projection: [`../configs/plan/excel-roadmap-v1.yaml`](../configs/plan/excel-roadmap-v1.yaml)
(regenerate with `python scripts/extract_excel_roadmap.py`; a test fails if it goes
stale).

Implementation contract, subordinate to the workbook and in agreement with it:
[`../configs/protocols/protocol-v2-domain-adaptive-fusion.yaml`](../configs/protocols/protocol-v2-domain-adaptive-fusion.yaml).

Operational pointer registry: [`../configs/active-state.yaml`](../configs/active-state.yaml).
It is a registry, not an amendment mechanism.

## 2. What is the objective?

Domain-Adaptive Fusion for zero-label cross-system log anomaly detection: learn from
several source architectures, adapt to an unseen target architecture using only an
unlabelled normal burn-in buffer, and detect anomalies with calibrated, explainable
evidence and a first-class abstain state.

## 3. Who owns decisions?

No agent owns scientific decisions or may amend the workbook without explicit researcher authorization. P2.0.1 is an explicitly authorized namespace migration. Dataset eligibility, buffer assumptions,
model selection, empirical runs and conclusions are researcher-owned. Gates G0–G4
are signed by the researcher, never by an agent.

## 4. What is active versus historical?

Active: the workbook, its projection, the v2 protocol contract, and the Phase 1
implementation under `src/seqlogad/protocol/`.

Historical provenance, controlling nothing: protocol v1.1 and its addenda
(`effect-001`, `split-clarification-v1`, `purge-decision-v1`), the HDFS/BGL five-way
split and sealed TEST artifacts, the frozen Drain3 parser state, the earlier master
plans under `Plan/`, and the retired task ladder
(`SEQ-001`, `THEORY-COMPLETE-001`, `KT-1/2/3`, `EFFECT-001`, the Transformer and
localization gates). See
[`../docs/migration/EXCEL-PLAN-MIGRATION-001.md`](../docs/migration/EXCEL-PLAN-MIGRATION-001.md).

## 5. What is complete and what is next?

Phase 1 (P1.1–P1.8) is **complete and signed**. All eight tasks are `DONE` and
**G0 = `PASSED`** (19/19 criteria), signed by the researcher on 2026-09-06. The four
signatures are in `../configs/protocols/g0-signatures.yaml` and are re-verified by
exact payload match on every build.

`NEXT_AUTHORIZED_TASK = P3.6` — Fusion baseline (FUS-BASE-001, `configs/protocols/fus-base-v1.yaml`), opened and
preregistered on 2026-11-28. P3.5 is Done
(UNC-CS-001, `configs/protocols/unc-cs-v1.yaml`; receipt `configs/protocols/p3.5-receipt-v1.yaml`, 2026-11-28). P3.2–P3.4 are Done,
accepted on 2026-11-28 (`configs/protocols/p3.2-p3.4-receipt-v1.yaml`): both fold memories are frozen (ADAPT-MEM-001),
calibrated (CAL-CS-001) and have drift references (DRIFT-CS-001). P3.1 (held-out check) is Done:
G2-HANDOFF-001 (`configs/protocols/p3.1-g2-handoff-v1.yaml`) verified the P2.7 package and the G2 record.
Phase 2 is complete on seed 42 and frozen as P2-FREEZE-001 (`configs/protocols/p2-freeze-v1.yaml`), accepted by
the researcher on 2026-11-22; G2 = PARTIAL. P3.3 and P3.4 were opened on 2026-11-27 to preregister
CAL-CS-001 (`configs/protocols/cal-cs-v1.yaml`) and DRIFT-CS-001 (`configs/protocols/drift-cs-v1.yaml`); both were built
on the frozen P3.2 memory on 2026-11-28. P3.7 stays closed (G2 = PARTIAL); P3.8 onwards wait for the researcher. Namespace P2_ARCH_V1 and historical
crosswalk: `configs/protocols/phase2-roadmap-migration-v1.yaml`. Phase-2 contract:
`configs/protocols/phase2-architecture-v1.yaml`; source-label C2 resolved (source validation labels only).

G1–G4 remain `NOT_PASSED`. `EXC-003` blocks G1 for the ARCH-HADOOP fold: its burn-in
buffer is `REJECTED` under `BUFFER-REJECT-001`, so that fold may not build target
memory, calibrate, fit drift or set abstention thresholds. Relaxing the readiness
minima is a forbidden remedy.

Scientific results: `NOT_RUN`. No model has been trained.

## 6. What must never happen?

Target anomaly labels may not reach training, adaptation, calibration, routing,
fusion fitting, model selection, normaliser fitting or parser fitting. In particular
a burn-in buffer may never be built by selecting records whose label is normal. The
only place ground truth may be opened is
[`../src/seqlogad/protocol/labels.py`](../src/seqlogad/protocol/labels.py), under an
explicit evaluation scope, and every access is recorded.
