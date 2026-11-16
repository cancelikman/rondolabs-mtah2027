# Per-match report (rondo_r7)

| match | spellF1 | macroF1 | possessions true / pred | recall <1s | 1-2s | 2-5s | 5-10s | >10s | wrong frames: ball missing | contested | loose | top error type (gain) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SC2006229 | 46.4 | 87.1 | 317 / 229 | 0% | 3% | 20% | 52% | 72% | 19% | 12% | 22% | missed:wrong_team (+19.5) |
| SC2006363 | 47.6 | 87.3 | 337 / 259 | 0% | 6% | 32% | 54% | 67% | 17% | 16% | 20% | missed:short<2s (+19.5) |
| SC2007448 | 46.6 | 86.0 | 272 / 236 | 0% | 10% | 30% | 52% | 69% | 20% | 12% | 23% | missed:wrong_team (+16.2) |
| SC2007721 | 49.3 | 86.6 | 308 / 253 | 0% | 13% | 27% | 53% | 73% | 19% | 12% | 23% | missed:wrong_team (+18.1) |
| SC2010085 | 43.8 | 86.0 | 279 / 233 | 7% | 14% | 30% | 35% | 59% | 18% | 12% | 24% | spurious:split_piece (+20.1) |
| SC1959846 | 44.5 | 85.8 | 356 / 255 | 0% | 6% | 25% | 57% | 75% | 15% | 15% | 23% | missed:short<2s (+17.6) |
| SC1986691 | 50.6 | 84.7 | 287 / 245 | 0% | 5% | 34% | 61% | 72% | 34% | 12% | 17% | missed:short<2s (+15.5) |

**All matches**

Recall of true possessions by duration: <1s 1% (n=158), 1-2s 8% (n=317), 2-5s 28% (n=461), 5-10s 52% (n=448), >10s 69% (n=772)

Where the wrong frames are (mean share): ball missing 20%, contested (both teams within 2 m) 13%, loose (nobody within 3 m, ball visible) 22%, other 45%

| error type | mean spellF1 gain if fixed | matches |
|---|---|---|
| missed:wrong_team | +17.2 | 7 |
| missed:short<2s | +17.1 | 7 |
| spurious:in_other_team | +13.9 | 7 |
| spurious:split_piece | +13.7 | 7 |
| missed:split | +4.2 | 7 |
| missed:called_dead | +3.6 | 7 |
| missed:boundary | +1.9 | 7 |
| spurious:in_dead | +1.4 | 7 |
| missed:swallowed | +0.3 | 7 |
