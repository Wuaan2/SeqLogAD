# C2 source-label scope amendment — 2026-11-22

Status: decided by the researcher on 2026-11-22 (option A), before any P2.6 scoring pass and before any
expert score on the P2.6 evaluation units was computed or seen.
Config: `configs/protocols/c2-source-label-scope-v1.yaml`.
Resolves: blocker C2 (`source_label_blocker` in `configs/protocols/phase2-architecture-v1.yaml`) for the
uses listed below. The frozen Phase-1/P2.0 contract files keep their historical `UNRESOLVED_DENY` values;
this amendment and its config are the authority for the new scope, as the P3.1 amendment was for
checkpoint selection.

## 1. Decision

Public Loghub anomaly labels of **source** architectures may be used for **P2.6 supervised source
evidence** (AP, AUROC, false-alarm rate, double-fault, incremental evidence of equal-weight combinations),
in addition to the checkpoint selection already authorised by the P3.1 amendment (2026-09-27).

Wording for reports: *label-free source training with labeled source development and labeled source
diversity evaluation*. Target labels remain unused until P4.

## 2. Scope

| Item | Rule |
|---|---|
| Architectures | ARCH-HDFS (block units) and ARCH-BGL (log-line units), only while they are a **source** of the fold |
| Excluded | ARCH-HADOOP (labels unreliable, loghub #56; EXC-007) and ARCH-OPENSTACK (validation lacks both classes; EXC-004) |
| Partition | `SOURCE_VALIDATION` only |
| Label file | the existing sidecar `phase2-source-development-v13r1` (manifest `885b5382…`); no new label file is read |
| Units | the P2.6 evaluation units defined in the P2.6 preregistration: validation windows **not** used for checkpoint selection |

## 3. Forbidden

- Any target-architecture label, `TARGET_BURN_IN` or `TARGET_EVALUATION` data.
- Labels in model inputs, features, prompts or training losses.
- Re-selecting checkpoints, re-training, or changing a frozen expert config after seeing P2.6 results.
- Using P2.6 results to set fusion weights, gate parameters, thresholds or calibration for P3. The P3 gate
  is trained only by its own preregistered source-episodic procedure.
- Signing G2 by an agent. P2.6 reports evidence and a recommended verdict; the researcher signs.

## 4. Consequences

- `G2_REQUIRED_LABEL_BASED_EVIDENCE` can be produced for HDFS and BGL as sources.
- Each fold has exactly one labeled source (BGL fold: HDFS; HDFS fold: BGL), so evidence is per fold and
  statistical power is low; every number is reported with a unit-level bootstrap interval.
- Hadoop and OpenStack enter P2.6 through label-free diagnostics only.
