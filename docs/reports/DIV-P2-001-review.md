# DIV-P2-001 — post-hoc review before G2 sign-off (2026-11-22)

Sensitivity checks on the HDFS fold (labeled source ARCH-BGL, 1,536 lines, 67 positives). Not preregistered;
the preregistered verdicts in `DIV-P2-001.md` are unchanged. Script: read-only, no label beyond the C2 scope.

| Check | Finding |
|---|---|
| Duplicated units | 67 positives are only 29 distinct target messages; 27 are one message (`rts: kernel terminated for reason <NUM>`) with identical context, so identical scores. P2.2/P2.3 give 373 distinct scores for 1,536 lines. Units are not independent. |
| Labels follow the message | No negative line shares a positive line's message. |
| LOW_COVERAGE windows | 23 units, no positive; excluding them: P2.2 0.196, P2.3 0.171, P2.2+P2.3 0.435 AP. No effect. |
| Dominant message | Without it: P2.2 0.107, P2.3 0.179, P2.2+P2.3 0.502 AP. The P2.2+P2.3 gain does not come from that message. |
| Mechanism | On positives the two experts' ranks are almost unrelated (Spearman 0.10; negatives 0.68). Of 120 negatives above the 95th percentile in either expert, only 1 is high in both, so averaging ranks removes expert-specific false alarms. |
| Combination rule | Mean percentile (preregistered) 0.417, min percentile 0.422, mean z-score 0.160, max percentile 0.139 AP. The gain needs a rank-based rule; z-scores fail because raw NLL tails are heavy. |
| Dependence-aware interval | Bootstrap over 174 message clusters instead of lines: P2.2+P2.3 ΔAP +0.239, 95% [0.088, 0.556] (still > 0). P2.2+GTAT ΔAP +0.083, 95% [-0.023, 0.214] (includes 0). |

Reading: the P2.3 addition in the HDFS fold survives every check except a z-score combination. The GTAT addition
is not robust once duplicated lines are clustered. Neither changes the recommended G2 = PARTIAL, which already
requires agreement across both folds; the BGL fold (20 positive HDFS blocks) shows no added evidence.
