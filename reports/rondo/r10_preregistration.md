# Round 10 pre-registration (written 2026-11-16 17:00 Istanbul, before any result on the new matches)

New data: SkillCorner open data now has 20 matches. The 10 never used by the team
(SC2006229, SC2006363, SC2007448, SC2007721, SC2010085, SC2011166, SC2013725, SC2015213,
SC2016236, SC2017461) form a fresh, independent test set "N10". Labels: merge<0.5s, as the twin.
Only SC2006229 had been scored (F4 0.4265) when this note was written; no F6 number on N10 was seen.

R10-1 (submission check, no training): F6 (models/rondo_r7) vs F4 (models/rondo_r5) on N10.
  Rule: keep F6 unless mean spellF1 delta (F6 - F4) < -1.0 point; then switch to F4.
  Report also: per-match deltas, wins/losses, paired bootstrap over matches.

R10-2 (only if time allows): any new candidate is selected on N10 split
  N10-A = first 5 ids sorted (training/selection), N10-B = last 5 (test, opened once).
  Accept only if: N10-B mean spellF1 >= F6 + 1.5 AND guards (WOY, MT1, MT2, Groningen H2) lose <= 1.

## Addendum 2026-11-16 18:20 (before any R10-2 number)
Candidates for R10-2 are fixed in rondo/r10.py:
  C0 = F6 recipe, correction model refit on all 10 old twin matches + 5 IDSSE (data only).
  C1 = C0 + control-memory features (rondo/control_memory.py) + agreement column.
Control-memory settings fixed before N10-A: radius 1.5 m, relative speed 4 m/s, min run 0.3 s,
contest 2 m (chosen from a sanity check on SC1959846 only: last-control agreement with the twin
label 84.9% vs nearest-team 75.2% on ball-visible live frames). Alpha 0.7, team cost 6 unchanged.
No other setting is tuned.

## Addendum 2026-11-16 20:10: E2 (decoder control-start evidence), stopped before N10-A
Sanity check on SC1959846 (a training match, not a selection set): F6 44.52; raising P(new team)
at control-run starts gives 44.83 (strength 0.3 and 0.5), 44.59 (0.7), 44.11 (0.9), 43.68 (1.0).
The event model already gives P(new H/A) ~0.13 on average, so the extra evidence adds little.
Expected N10-A gain far below +1.5: E2 not run on N10-A; N10-B stays closed. F6 remains the submission.

## Addendum R11, 2026-11-16 21:20 (before any R11 number on N10-A, guards or N10-B)
Motivation (diagnostic on N10-A + 2 development matches, tools/match_report.py): recall of true
possessions by duration <1 s 1%, 1-2 s 8%, 2-5 s 28%, 5-10 s 52%, >10 s 69%; the four largest error
types are all about where the team switches happen.
Candidate: a possession-change detector (rondo/r11.py). Target and class balancing identical to
the existing transition head (event_targets, radius 2 frames). Inputs: F6 final posteriors, F6
event probabilities, stack geometry/window features, control-memory features. Data: 10 old twin
matches + 5 IDSSE (clean, target-like), equal weight per source, swap-symmetric. Used as
event' = w*detector + (1-w)*F6 event, w in {0.5, 1.0}; F6 decoder unchanged.
Rule: N10-A mean spellF1 >= F6 + 1.5 and every guard >= -1.0 vs F6; then N10-B once, new
package iff mean delta >= +1.5. A plumbing check trains on SC1959846 and runs on SC1986691
(both old development matches, already in the training set of R11) before the real run.

## R11 result (2026-11-16 21:37): rejected
N10-A: w 0.5 +0.10 (2/5 wins), w 1.0 +0.08; worst guard -1.36 / -3.35 (MT2 clean). N10-B stays closed.

## Addendum R12, 2026-11-16 22:00 (before any R12 number on N10-A, guards or N10-B)
Candidate: set/graph model over ball + all players (rondo/gnn.py: per-player relative position,
velocity, relative speed, pitch position; attention + mean pooling per team; 21-step window of
0.2 s; dilated temporal convolutions; H/A/D). Trained from scratch on 10 old twin matches + 5 IDSSE
(clean, target-like), equal samples per source, random team swap, seed 20261116, 4 x 300k windows.
Use: state = (1-w)*F6 final posteriors + w*GNN, w in {0.25, 0.5}; F6 events and decoder unchanged.
Rule: N10-A mean spellF1 >= F6 + 1.5, every guard >= -1.0; then N10-B once, new package iff >= +1.5.
A cloud plumbing check (8 old twin matches -> SC1959846, SC1986691) runs in parallel; it does not
choose anything.

## R12 result on N10-A and guards (2026-11-16 22:46): rejected under the R12 rule
w 0.25: N10-A +0.20 (3/5), worst guard -2.66 (GRO clean). w 0.5: N10-A +0.78 (5/5), worst guard
-7.53 (GRO clean). Target-like guards with w 0.5: WOY +2.14, MT1 +2.27, MT2 +3.76, GRO -2.75;
clean guards: WOY -4.43, MT1 -1.25, MT2 -0.19, GRO -7.53.

## Addendum R12-B, 2026-11-16 22:55 (team decision, written before N10-B is opened)
The team accepts a higher-variance candidate because the competition tracking is noisy and the
GNN gained on N10-A (5/5) and on 3 of 4 target-like guards. This is a NEW rule, decided after
seeing the N10-A/guard numbers above; it is recorded as such.
Candidate: R12 GNN, w = 0.5 (fixed; no other value is tried).
Rule: open N10-B once. New package F7 iff mean spellF1 delta vs F6 >= +1.5 AND the candidate is
ahead in at least 4 of the 5 N10-B matches. The known clean-guard losses (up to -7.5 on Groningen
H2 clean) are accepted. Otherwise F6 stays. No further candidate will be tested on N10-B.

## R12-B result (2026-11-16 23:06): passed
N10-B: +1.58 mean spellF1 vs F6, 5/5 wins (+1.97, +2.00, +2.01, +1.02, +0.88), macroF1 +0.71.
F7 = models/rondo_r8, SHA-256 7f63884e1c1cdac70d953cc6832576e9f09498c74256ea2b87c8f8d68905bdad (two identical runs).
Public window, run once as a pipeline check after the decision: F7 58.7 / 84.8, F6 50.0 / 83.0.

## Addendum R13, 2026-11-16 23:40 (before any R13 number)
Seed ensemble of the R12 graph network: two more GNNs with the exact R12 recipe, seeds 20261117 and
20261118; F8 = F6 posteriors mixed at w = 0.5 with the mean of the three GNNs. No other setting.
Rule: adopt F8 iff its mean spellF1 over all 10 N10 matches >= F7's and every guard (clean and
target-like) is within 1.0 of F7. Otherwise F7 stays. (N10 has been opened; the rule is a no-worse
check for a change without new tunable settings.) Each seed's own delta vs F6 is reported to show
whether F7's gain depends on its seed.

## R13 result (2026-11-17 00:18): F7 stays
N10 (10 matches) mean spellF1: F6 45.37, F7 46.55, seed 20261117 46.49, seed 20261118 46.37,
three-seed mean (F8) 46.22. F8 < F7 on N10, so the rule keeps F7 (guards: F8 within -0.02 of F7).
Every seed alone gains +1.0 to +1.2 over F6: F7's gain does not depend on its seed.
