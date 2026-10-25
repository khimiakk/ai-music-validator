"""tempo and CLAP window embeddings for every track."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimv.audio.clap import DEFAULT_MODEL  # noqa: E402
from aimv.audio.io import load_audio, windows  # noqa: E402
from aimv.audio.tempo import estimate_tempo  # noqa: E402
from aimv.data import AUDIO_DIR, PROCESSED_DIR, RAW_CSV, audio_path, load_sunocaps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RAW_CSV))
    ap.add_argument("--audio-dir", default=str(AUDIO_DIR))
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--windows", type=int, default=3, help="evenly spaced 10 s windows; 0 = sliding")
    ap.add_argument("--hop", type=float, default=10.0, help="hop (s) when --windows 0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-clap", action="store_true")
    ap.add_argument("--no-tempo", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    feat_path, emb_path = out / "audio_features.csv", out / "clap_audio_windows.npz"

    df = load_sunocaps(args.csv)
    track_ids = df["track_id"].tolist()[: args.limit]

    feats = pd.read_csv(feat_path).set_index("track_id").to_dict("index") if feat_path.exists() else {}
    embs = dict(np.load(emb_path)) if emb_path.exists() else {}

    scorer = None
    if not args.no_clap:
        from aimv.audio.clap import ClapScorer

        print(f"Loading CLAP {args.model} ...", flush=True)
        scorer = ClapScorer(args.model)
        print(f"  device={scorer.device}, sr={scorer.sr}", flush=True)
        json.dump({"model": args.model, "logit_scale": scorer.logit_scale, "windows": args.windows,
                   "hop_s": args.hop, "win_s": 10.0}, open(out / "clap_meta.json", "w"), indent=2)
    sr = scorer.sr if scorer else 48_000

    missing, t0 = [], time.time()
    for i, tid in enumerate(track_ids, 1):
        need_tempo = not args.no_tempo and tid not in feats
        need_clap = scorer is not None and tid not in embs
        if not (need_tempo or need_clap):
            continue
        try:
            path = audio_path(tid, args.audio_dir)
        except FileNotFoundError:
            missing.append(tid)
            continue
        y = load_audio(path, sr=sr)
        row = feats.get(tid, {})
        row["duration_s"] = len(y) / sr
        if need_tempo:
            row.update(estimate_tempo(y, sr))
        feats[tid] = row
        if need_clap:
            ws = windows(y, sr, win_s=10.0, n_windows=args.windows or None, hop_s=args.hop)
            embs[tid] = scorer.embed_audio(ws)
        if i % 10 == 0 or i == len(track_ids):
            _save(feats, embs, feat_path, emb_path)
            el = time.time() - t0
            print(f"[{i}/{len(track_ids)}] {el / 60:.1f} min elapsed", flush=True)
    _save(feats, embs, feat_path, emb_path)
    print(f"Done. tempo rows={len(feats)}, clap tracks={len(embs)}")
    if missing:
        print(f"WARNING: no audio found for {len(missing)} tracks, e.g. {missing[:5]}"
              f" (looked for <track_id>.mp3 under {args.audio_dir})")


def _save(feats, embs, feat_path, emb_path):
    if feats:
        pd.DataFrame.from_dict(feats, orient="index").rename_axis("track_id").reset_index().to_csv(
            feat_path, index=False)
    if embs:
        np.savez(emb_path, **embs)


if __name__ == "__main__":
    main()
