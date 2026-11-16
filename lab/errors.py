"""SpellF1 kaybinin nereden geldigini cikarir.

Skorlarin kendisi resmi evaluation/possession_metrics.py ile hesaplanir; bu
dosya yalnizca "hangi sahiplik neden eslesmedi" sorusunu yanitlar.

Kodlar: H=0, A=1, D=2, X=3 (resmi skorlayiciyla ayni).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
IOUS = tuple(pm.SPELL_IOUS)
NAMES = "HADX"

KINDS = {
    "merge_dead": "Birleşme: aradaki ölü top kaçtı",
    "merge_opp": "Birleşme: aradaki rakip sahiplik kaçtı",
    "split_dead": "Bölünme: uydurma ölü top",
    "split_opp": "Bölünme: uydurma rakip sahiplik",
    "as_dead": "Kaçtı: ölü top sanıldı",
    "wrong_team": "Kaçtı: rakibin sahipliği sanıldı",
    "boundary_start": "Sınır kayması: başlangıç",
    "boundary_end": "Sınır kayması: bitiş",
    "boundary_both": "Sınır kayması: iki uç",
    "spurious_in_dead": "Uydurma sahiplik: ölü top içinde",
    "spurious_in_opp": "Uydurma sahiplik: rakip sahiplik içinde",
    "fragment": "Eşleşmeyen tahmin: aynı takımın sahipliği üstünde (artık parça ya da aşırı uzun)",
    "spurious_other": "Uydurma sahiplik: diğer",
}


def greedy_matches(gt_spells, pred_spells):
    """Her esik icin (gt_index -> pred_index) eslesmesi; resmi acgozlu kuralin aynisi."""
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    out = {}
    for threshold in IOUS:
        used_gt, used_pred, pairs = set(), set(), {}
        for iou, i, j in overlaps:
            if iou < threshold:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            pairs[i] = (j, iou)
        out[threshold] = pairs
    return out, overlaps


def _share(codes, start, stop):
    window = codes[start:stop]
    return [float(np.mean(window == k)) for k in (H, A, D)]


def _classify_gt(index, spell, gt, pred, gt_spells, pred_spells, best):
    """Eslesemeyen (ya da kismen eslesen) gercek sahipligin baskin nedeni."""
    start, stop, team = spell
    length = stop - start
    share = _share(pred, start, stop)
    partner = best.get(index)
    if partner is None or share[team] < .3:
        return ("as_dead" if share[D] >= share[1 - team] else "wrong_team"), {}
    j = partner
    p_start, p_stop, _ = pred_spells[j]
    # Tahmin, ayni takimin baska bir gercek sahipligini de yutuyor mu?
    swallowed = [g for k, g in enumerate(gt_spells)
                 if k != index and g[2] == team
                 and min(g[1], p_stop) - max(g[0], p_start) >= max(3, .5 * (g[1] - g[0]))]
    if swallowed:
        other = min(swallowed, key=lambda g: abs(g[0] - start))
        a, b = (stop, other[0]) if other[0] >= stop else (other[1], start)
        between = gt[a:b]
        dead = float(np.mean(between == D)) if len(between) else 0.
        return ("merge_dead" if dead >= .5 else "merge_opp"), {"gap_frames": int(b - a)}
    # Gercek sahiplik birden fazla ayni-takim tahminine mi bolunmus?
    pieces = [p for p in pred_spells if p[2] == team
              and min(p[1], stop) - max(p[0], start) >= max(3, .15 * length)]
    if len(pieces) >= 2:
        pieces.sort()
        between = pred[pieces[0][1]:pieces[1][0]]
        dead = float(np.mean(between == D)) if len(between) else 0.
        return ("split_dead" if dead >= .5 else "split_opp"), {"pieces": len(pieces)}
    d_start, d_stop = p_start - start, p_stop - stop
    tolerance = max(2, .1 * length)
    info = {"start_error": int(d_start), "end_error": int(d_stop)}
    if abs(d_start) > tolerance and abs(d_stop) > tolerance:
        return "boundary_both", info
    return ("boundary_start" if abs(d_start) >= abs(d_stop) else "boundary_end"), info


def _classify_pred(spell, gt):
    start, stop, team = spell
    share = _share(gt, start, stop)
    if share[D] >= .5:
        return "spurious_in_dead"
    if share[1 - team] >= .5:
        return "spurious_in_opp"
    if share[team] >= .5:
        return "fragment"
    return "spurious_other"


def analyse(gt, pred):
    """gt, pred: ayni uzunlukta kod dizileri. Skorlar + hata listesi dondurur."""
    gt = np.asarray(gt, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    result = pm.evaluate(gt, pred)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    matches, overlaps = greedy_matches(gt_spells, pred_spells)

    best = {}
    for iou, i, j in overlaps:  # IoU azalan sirada: ilk gorulen en iyisi
        best.setdefault(i, j)
    gt_hits = np.array([sum(i in matches[t] for t in IOUS) for i in range(len(gt_spells))], dtype=int)
    matched_pred = {t: {j for j, _ in matches[t].values()} for t in IOUS}
    pred_hits = np.array([sum(j in matched_pred[t] for t in IOUS) for j in range(len(pred_spells))], dtype=int)

    errors = []
    for i, spell in enumerate(gt_spells):
        if gt_hits[i] == len(IOUS):
            continue
        kind, info = _classify_gt(i, spell, gt, filled, gt_spells, pred_spells, best)
        errors.append(dict(side="gt", kind=kind, start=int(spell[0]), stop=int(spell[1]),
                           team=NAMES[spell[2]], hits=int(gt_hits[i]),
                           loss=round(float(1 - gt_hits[i] / len(IOUS)), 3), **info))
    for j, spell in enumerate(pred_spells):
        if pred_hits[j] > 0:
            continue  # kismen eslesen tahminin kaybi zaten gercek tarafta sayildi
        errors.append(dict(side="pred", kind=_classify_pred(spell, gt), start=int(spell[0]),
                           stop=int(spell[1]), team=NAMES[spell[2]], hits=0, loss=1.))
    errors.sort(key=lambda e: (-e["loss"], e["start"]))
    for k, error in enumerate(errors):
        error["id"] = k

    taxonomy = {}
    for error in errors:
        row = taxonomy.setdefault(error["kind"], dict(key=error["kind"], label=KINDS[error["kind"]],
                                                      count=0, loss=0.))
        row["count"] += 1
        row["loss"] += error["loss"]
    taxonomy = sorted(taxonomy.values(), key=lambda r: -r["loss"])
    for row in taxonomy:
        row["loss"] = round(float(row["loss"]), 2)

    fps = pm.FPS
    edges = [(0, 1, "< 1 sn"), (1, 3, "1-3 sn"), (3, 10, "3-10 sn"), (10, 1e9, "> 10 sn")]

    def buckets(spells, hits):
        rows = []
        for low, high, label in edges:
            pick = [k for k, s in enumerate(spells) if low <= (s[1] - s[0]) / fps < high]
            rows.append(dict(label=label, n=len(pick),
                             rate=round(float(np.mean(hits[pick]) / len(IOUS)), 3) if pick else None))
        return rows

    return dict(
        scores=dict(
            spell=result["spell_f1_mean"], macro=result["macro_f1"],
            n_gt=result["n_gt_spells"], n_pred=result["n_pred_spells"],
            dur_gt=result["gt_mean_spell_duration_s"], dur_pred=result["mean_spell_duration_s"],
            alive_dead_f1=result["alive_dead_f1"], team_acc=result["team_acc_gt_alive"],
            per_iou=[dict(iou=t, **{k: result["spell_f1"][t][k] for k in ("precision", "recall", "f1")})
                     for t in IOUS],
            per_class={k: v["f1"] for k, v in result["per_class"].items()}),
        runs=[[int(a), int(b)] for a, b in runs],
        gt_segments=[[int(a), int(b), int(c)] for a, b, c in pm.segments(gt, runs)],
        pred_segments=[[int(a), int(b), int(c)] for a, b, c in pm.segments(filled, runs)],
        errors=errors, taxonomy=taxonomy,
        gt_buckets=buckets(gt_spells, gt_hits), pred_buckets=buckets(pred_spells, pred_hits),
        oracles=oracles(gt, filled, runs, result["spell_f1_mean"]),
    )


def _fill_team(codes, mask, runs):
    """mask ile isaretli kareleri en yakin onceki (yoksa sonraki) takim etiketiyle doldurur."""
    out = codes.copy()
    out[mask] = X
    for start, stop in runs:
        window = out[start:stop]
        known = np.flatnonzero((window == H) | (window == A))
        if not len(known):
            continue
        is_team = (window == H) | (window == A)
        carry = np.maximum.accumulate(np.where(is_team, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        todo = window == X
        window[todo] = window[carry][todo]
    return out


def oracles(gt, pred, runs, base):
    """'Su parca kusursuz olsaydi skor ne olurdu' sorulari. Yalnizca teshis icindir."""
    graded = gt != X
    rows = []

    # 1) Olu top maskesi gercekten alinir, sahiplik bizim tahminimiz kalir.
    dead_fixed = pred.copy()
    dead_fixed = _fill_team(dead_fixed, graded & (gt != D) & (pred == D), runs)
    dead_fixed[graded & (gt == D)] = D
    rows.append(("Ölü top kusursuz olsaydı", dead_fixed,
                 "D sınırları gerçekten alındı, takım kararı bizim"))

    # 2) Bizim olu top maskemiz kalir, canli karelerde takim gercekten alinir.
    team_fixed = pred.copy()
    both = graded & (gt != D) & (pred != D)
    team_fixed[both] = gt[both]
    rows.append(("Takım kararı kusursuz olsaydı", team_fixed,
                 "D maskesi bizim, canlı karelerde takım gerçekten alındı"))

    # 3) 1 saniyeden kisa gercek sahiplikler yok sayilsaydi (komsusuna katilir).
    relaxed = gt.copy()
    for start, stop, code in pm.possession_spells(gt, runs):
        if stop - start < pm.FPS:
            before = gt[start - 1] if start > 0 and gt[start - 1] != X else None
            relaxed[start:stop] = before if before is not None else gt[min(stop, len(gt) - 1)]
    short_free = pm.evaluate(relaxed, pred)["spell_f1_mean"]

    out = [dict(name=name, spell=float(pm.evaluate(gt, fixed)["spell_f1_mean"]), note=note)
           for name, fixed, note in rows]
    out.append(dict(name="1 sn'den kısa gerçek sahiplikler olmasaydı", spell=float(short_free),
                    note="etiket tanımı farkının tavanı: kısa dönüşler komşusuna katıldı"))
    for row in out:
        row["delta"] = round(row["spell"] - base, 4)
    return out
