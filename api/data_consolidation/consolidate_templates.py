#!/usr/bin/env python3
"""
Consolidate all training-data sources under /templates into ONE unified base:

    templates/
      img/          renamed raw originals  -> <uid>.<ext>
      labels.csv    one row per image, unified schema
      old/          the original source folders (read from here)

All sources score the same standardized OCS-Plus figure: 20 elements x 3 binary
aspects (PRES/ACC/POS) = 60, summing to TotalScore 0-60 (see
templates/old/bsp_ocsplus_202511/readme.txt).

Pure stdlib, no image processing, no DB. Runs on the HOST (templates is
host-writable; the container mount is read-only). Idempotent: rebuilds img/ and
labels.csv from old/ on every run.

Usage:
    python3 api/data_consolidation/consolidate_templates.py \
        --templates /home/ste/NPSketch/templates
"""
import argparse
import csv
import os
import shutil
import sys

# ---- unified schema -------------------------------------------------------

ELEM_COLS = []
for n in range(1, 21):
    ELEM_COLS += [f"ELEM{n:02d}PRES", f"ELEM{n:02d}ACC", f"ELEM{n:02d}POS"]

COLUMNS = (
    ["uid", "patient_id", "source", "cond", "filename",
     "orig_id", "orig_filename", "total_score", "total_score_sum", "label_status"]
    + ELEM_COLS
    + ["gender", "age", "test_date"]
)

COND_MAP = {
    "fc0": "COPY", "fc1": "RECALL",
    "CPY": "COPY", "MEM": "RECALL",
    "COPY": "COPY", "RECALL": "RECALL",
}


def blank_row():
    return {c: "" for c in COLUMNS}


def sublabels_sum(row):
    """Sum the 60 element columns if all are filled, else None."""
    vals = []
    for c in ELEM_COLS:
        v = row[c]
        if v == "":
            return None
        vals.append(int(v))
    return sum(vals)


def finalize(row):
    """Fill total_score_sum and label_status."""
    s = sublabels_sum(row)
    row["total_score_sum"] = "" if s is None else str(s)
    ts = row["total_score"]
    try:
        tsv = int(ts) if ts != "" else (s if s is not None else None)
    except ValueError:
        tsv = None
    row["label_status"] = "zero" if (tsv is not None and tsv == 0) else "scored"
    return row


# ---- per-source loaders ---------------------------------------------------
# Each loader yields (unified_row, src_image_path) and appends skip reasons.


def load_telefred(old, deliveries, skips):
    """TeleFred deliveries as a union; earlier delivery wins on duplicate uid."""
    seen = set()
    for delivery in deliveries:
        base = os.path.join(old, delivery)
        if not os.path.isdir(base):
            continue
        for folder, cond in (("fc0", "COPY"), ("fc1", "RECALL")):
            csv_path = os.path.join(base, f"{folder}.csv")
            img_dir = os.path.join(base, folder)
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
                for r in csv.DictReader(f, delimiter=";"):
                    fname = r.get("FileName")
                    if not fname:
                        continue
                    stem = fname[:-4] if fname.lower().endswith(".png") else fname
                    # strip trailing -fc0/-fc1, replace with normalized cond
                    key = stem.rsplit("-", 1)[0]
                    uid = f"TF-{key}-{cond}"
                    if uid in seen:
                        continue  # later delivery duplicate -> precedence to earlier
                    img_path = os.path.join(img_dir, fname)
                    if not os.path.exists(img_path):
                        skips.append(("TELEFRED", uid, f"image missing: {fname}"))
                        continue
                    row = blank_row()
                    row.update(
                        uid=uid, patient_id=f"TF-{r['ID']}", source="TELEFRED",
                        cond=cond, orig_id=r["ID"], orig_filename=fname,
                        total_score=r.get("TotalScore", ""),
                        gender=r.get("Gender", ""), age=r.get("Age", ""),
                        test_date=r.get("TestDate", ""),
                    )
                    for n in range(1, 21):
                        row[f"ELEM{n:02d}PRES"] = r.get(f"Presence{n}", "")
                        row[f"ELEM{n:02d}ACC"] = r.get(f"Accuracy{n}", "")
                        row[f"ELEM{n:02d}POS"] = r.get(f"Position{n}", "")
                    seen.add(uid)
                    yield finalize(row), img_path


def load_algorithm(old, skips):
    base = os.path.join(old, "algorithm_training_data_20260112")
    csv_path = os.path.join(base, "FIGURECOPY_ALGORITHM_SCORED_2025-01-08.csv")
    img_dir = os.path.join(base, "imgs")
    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f, delimiter=";"):
            cond = COND_MAP.get(r["Cond"])
            if cond is None:
                continue
            fname = f"{r['ID']}_{cond}.png"
            img_path = os.path.join(img_dir, fname)
            uid = f"ALG-{r['ID']}-{cond}"
            if not os.path.exists(img_path):
                skips.append(("ALGORITHM", uid, f"image missing: {fname}"))
                continue
            row = blank_row()
            row.update(
                uid=uid, patient_id=f"ALG-{r['ID']}", source="ALGORITHM",
                cond=cond, orig_id=r["ID"], orig_filename=fname,
                total_score=r.get("TotalScore", ""),
            )
            for c in ELEM_COLS:
                row[c] = r.get(c, "")
            yield finalize(row), img_path


def load_ocs_machine(old, skips):
    base = os.path.join(old, "bsp_ocsplus_202511", "Machine_rater")
    img_dir = os.path.join(base, "imgs")
    for csv_name, cond in (("FigCopy_PCS_MACHINE_RATER_COPY.csv", "COPY"),
                           ("FigCopy_PCS_MACHINE_RATER_RECALL.csv", "RECALL")):
        csv_path = os.path.join(base, "ratings", csv_name)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
            for r in csv.DictReader(f, delimiter=","):
                fname = f"{r['ID']}_{cond}.jpg"
                img_path = os.path.join(img_dir, fname)
                uid = f"OCSM-{r['ID']}-{cond}"
                if not os.path.exists(img_path):
                    skips.append(("OCS_MACHINE", uid, f"image missing: {fname}"))
                    continue
                row = blank_row()
                row.update(
                    uid=uid, patient_id=f"OCSM-{r['ID']}", source="OCS_MACHINE",
                    cond=cond, orig_id=r["ID"], orig_filename=fname,
                    total_score="",  # no TotalScore column -> use sum
                )
                for c in ELEM_COLS:
                    row[c] = r.get(c, "")
                s = sublabels_sum(row)
                if s is not None:
                    row["total_score"] = str(s)
                yield finalize(row), img_path


def load_oxford(old, skips):
    base = os.path.join(old, "training_data_oxford_manual_rater_202512")
    csv_path = os.path.join(base, "Rater1_simple.csv")
    img_dir = os.path.join(base, "imgs")
    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f, delimiter=","):
            cond = COND_MAP.get(r["Cond"])
            if cond is None:
                continue
            fname = f"{r['ID']}_{cond}.png"
            img_path = os.path.join(img_dir, fname)
            uid = f"OXF-{r['ID']}-{cond}"
            if not os.path.exists(img_path):
                skips.append(("OXFORD", uid, f"image missing: {fname}"))
                continue
            row = blank_row()
            row.update(
                uid=uid, patient_id=f"OXF-{r['ID']}", source="OXFORD",
                cond=cond, orig_id=r["ID"], orig_filename=fname,
                total_score=r.get("TotalScore", ""),
            )
            yield finalize(row), img_path  # no sublabels


def _split_elem_score(v):
    """LOWSCORER element score (presence+accuracy+position summed) -> (PRES, ACC, POS).

    The manual rating gives one 0-3 value per element. Resolve to the 3 binary
    aspects: ''/0 -> absent; 1 -> presence only; 3 -> all three; 2 -> presence+
    accuracy (chosen via presence/accuracy correlation; model-based disambiguation
    was inconclusive on these sparse low-score images — see CHANGELOG/branch notes).
    Sum is preserved (== the element score).
    """
    v = (v or "").strip()
    if v in ("", "0"):
        return 0, 0, 0
    return {1: (1, 0, 0), 2: (1, 1, 0), 3: (1, 1, 1)}[int(v)]


def load_lowscorer(old, skips):
    """Manually-rated low-score figures (1-15 of 60), extra training data.

    Source: old/low_scores/Bildbewertung.csv (20 ELEM scores + TotalScore, one
    row per image, image_path -> jpg). Standalone copy figures, no COPY/RECALL
    pairing -> cond='MANUAL', patient_id == uid.
    """
    base = os.path.join(old, "low_scores")
    csv_path = os.path.join(base, "Bildbewertung.csv")
    if not os.path.exists(csv_path):
        return
    with open(csv_path, encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f):
            rel = (r.get("image_path") or "").strip()
            if not rel:
                continue
            stem = os.path.splitext(os.path.basename(rel))[0]  # "15-Punkte-01"
            uid = f"LS-{stem}"
            img_path = os.path.join(base, rel)
            if not os.path.exists(img_path):
                skips.append(("LOWSCORER", uid, f"image missing: {rel}"))
                continue
            row = blank_row()
            row.update(
                uid=uid, patient_id=uid, source="LOWSCORER", cond="MANUAL",
                orig_id=stem, orig_filename=os.path.basename(rel),
                total_score=(r.get("Figure Copy - TotalScore") or "").strip(),
            )
            for n in range(1, 21):
                pres, acc, pos = _split_elem_score(r.get(f"ELEM{n:02d}"))
                row[f"ELEM{n:02d}PRES"] = str(pres)
                row[f"ELEM{n:02d}ACC"] = str(acc)
                row[f"ELEM{n:02d}POS"] = str(pos)
            yield finalize(row), img_path


# ---- driver ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default="/home/ste/NPSketch/templates")
    ap.add_argument("--dry-run", action="store_true",
                    help="build labels.csv + report but do not copy images")
    args = ap.parse_args()

    templates = os.path.abspath(args.templates)
    old = os.path.join(templates, "old")
    img_out = os.path.join(templates, "img")
    csv_out = os.path.join(templates, "labels.csv")

    if not os.path.isdir(old):
        sys.exit(f"ERROR: {old} not found. Move source folders into old/ first.")

    skips = []
    loaders = [
        ("TELEFRED", load_telefred(old, ["training_data_telefred_202606",
                                         "training_data_telefred_20260119"], skips)),
        ("ALGORITHM", load_algorithm(old, skips)),
        ("OCS_MACHINE", load_ocs_machine(old, skips)),
        ("OXFORD", load_oxford(old, skips)),
        ("LOWSCORER", load_lowscorer(old, skips)),
    ]

    rows = []
    copies = []  # (src_path, dst_path)
    per_source = {}
    uids = {}          # uid -> (source, row)
    dup_skips = []     # (source, uid, note)
    for src_name, gen in loaders:
        n = 0
        for row, img_path in gen:
            uid = row["uid"]
            if uid in uids:
                prev_src, prev_row = uids[uid]
                if prev_src != src_name:
                    # source prefixes should make this impossible
                    sys.exit(f"FATAL: cross-source uid collision '{uid}' "
                             f"({src_name} vs {prev_src})")
                # within-source duplicate (e.g. repeated CSV row) -> keep first
                note = "duplicate row"
                if prev_row["total_score"] != row["total_score"]:
                    note = (f"duplicate row WITH DIFFERENT score "
                            f"({prev_row['total_score']} kept, {row['total_score']} dropped)")
                dup_skips.append((src_name, uid, note))
                continue
            uids[uid] = (src_name, row)
            ext = os.path.splitext(img_path)[1].lower()
            row["filename"] = uid + ext
            rows.append(row)
            copies.append((img_path, os.path.join(img_out, uid + ext)))
            n += 1
        per_source[src_name] = n

    # write CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # copy images
    if not args.dry_run:
        if os.path.isdir(img_out):
            shutil.rmtree(img_out)
        os.makedirs(img_out)
        for src, dst in copies:
            shutil.copy2(src, dst)

    # report
    print("=" * 60)
    print("CONSOLIDATION REPORT")
    print("=" * 60)
    total = 0
    for src, n in per_source.items():
        print(f"  {src:13} {n:5d} rows")
        total += n
    print(f"  {'TOTAL':13} {total:5d} rows")
    print(f"\nimages {'(dry-run, not copied)' if args.dry_run else 'copied'}: {len(copies)}")
    print(f"labels.csv: {csv_out}")

    # QA: total_score vs sublabel sum
    print("\nLabel QA (total_score vs sublabel sum):")
    for src in per_source:
        srows = [r for r in rows if r["source"] == src]
        with_sub = [r for r in srows if r["total_score_sum"] != "" and r["total_score"] != ""]
        if not with_sub:
            print(f"  {src:13} no sublabels")
            continue
        exact = sum(1 for r in with_sub if r["total_score"] == r["total_score_sum"])
        gross = [r["uid"] for r in with_sub
                 if abs(int(r["total_score"]) - int(r["total_score_sum"])) > 5]
        print(f"  {src:13} {exact}/{len(with_sub)} exact; gross (|d|>5): {len(gross)}")

    # cond distribution
    print("\nCondition distribution:")
    for src in per_source:
        c = sum(1 for r in rows if r["source"] == src and r["cond"] == "COPY")
        rc = sum(1 for r in rows if r["source"] == src and r["cond"] == "RECALL")
        print(f"  {src:13} COPY={c} RECALL={rc}")

    # skips
    print(f"\nskipped (label without image): {len(skips)}")
    for s in skips[:10]:
        print("   ", s)
    if len(skips) > 10:
        print(f"    ... +{len(skips) - 10} more")

    # within-source duplicate rows (kept first)
    print(f"\nduplicate rows dropped (kept first): {len(dup_skips)}")
    for s in dup_skips[:10]:
        print("   ", s)

    # patient grouping sanity
    pats = {r["patient_id"] for r in rows}
    print(f"\ndistinct patients: {len(pats)} | rows: {len(rows)}")


if __name__ == "__main__":
    main()
