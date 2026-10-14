#!/usr/bin/env python3
"""
Quantifies trajectory SHAPE (not velocity) per batch, per phase (PRE/POST), from the same
center_x/center_y per-frame data already used for velocity, with the same held-frame exclusion
already validated (load_all_batch_frames, exclude_held=True -- see search.py docstring).

Two metrics, computed on each phase's ordered (center_x, center_y) points:
  straightness = net_displacement / total_path_length   in (0, 1], 1 = perfectly straight line,
                 near 0 = the path loops back near its own start (classic tortuosity/sinuosity
                 measure). Undefined (NaN) if total_path_length ~ 0 (no real movement) or fewer
                 than 3 points.
  loop_count   = (sum of absolute turning-angle changes between consecutive path segments) /
                 (2*pi) -- a standard proxy for "how many full rotations worth of turning" the
                 path contains; a straight line scores ~0, a path that circles back on itself
                 once scores ~1, etc. Requires >= 3 points (2 segments minimum).

Usage:
    python compute_trajectory_shape_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '../interval_selection')
import search as _search
_search.CSV_DIR = '../tracking_results_sam_dense2/csv'
from search import load_all_batch_frames


def smooth(xy, window=5):
    """Rolling-mean smoothing on each coordinate, edge-padded, before shape metrics -- raw
    frame-to-frame steps are dominated by sub-pixel/tracking jitter noise (verified: unsmoothed
    loop_count came out at 30-100+ 'loops' per phase, wildly inconsistent with the handful of
    clean, large-scale loops actually visible in the trajectory plots). Smoothing over a short
    window suppresses that high-frequency noise while preserving genuine larger-scale path
    structure (loops spanning many frames survive; single-frame jitter reversals don't)."""
    n = len(xy)
    if n < window:
        return xy
    kernel = np.ones(window) / window
    pad = window // 2
    xy_pad = np.pad(xy, ((pad, pad), (0, 0)), mode='edge')
    sx = np.convolve(xy_pad[:, 0], kernel, mode='valid')
    sy = np.convolve(xy_pad[:, 1], kernel, mode='valid')
    return np.column_stack([sx[:n], sy[:n]])


def straightness_and_loops(xy_raw):
    """xy_raw: (n,2) array of (x,y) points, ordered in time. Returns (straightness, loop_count, n)."""
    n = len(xy_raw)
    if n < 2:
        return np.nan, np.nan, n
    xy = smooth(xy_raw, window=5)
    seg = np.diff(xy, axis=0)  # (n-1, 2)
    seg_len = np.linalg.norm(seg, axis=1)
    total_path_length = seg_len.sum()
    net_displacement = np.linalg.norm(xy[-1] - xy[0])
    straightness = net_displacement / total_path_length if total_path_length > 1e-9 else np.nan

    if n < 3:
        return straightness, np.nan, n
    # turning angle between consecutive segment vectors, only where both segments have nonzero length
    valid = (seg_len[:-1] > 1e-9) & (seg_len[1:] > 1e-9)
    ang1 = np.arctan2(seg[:-1, 1], seg[:-1, 0])
    ang2 = np.arctan2(seg[1:, 1], seg[1:, 0])
    dtheta = ang2 - ang1
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    loop_count = np.abs(dtheta[valid]).sum() / (2 * np.pi) if valid.sum() > 0 else np.nan
    return straightness, loop_count, n


def main():
    frames = load_all_batch_frames(exclude_held=True)
    traj_ids = sorted(int(x.strip()) for x in open('trajectory_descriptions_dense2/all_trajectory_batch_ids.txt'))

    rows = []
    for bid in traj_ids:
        df = frames.get(bid)
        if df is None:
            rows.append({'batch_id': bid, 'note': 'no_csv'})
            continue
        row = {'batch_id': bid}
        for phase in ['pre', 'post']:
            sub = df[df['phase'] == phase].sort_values('frame_index')
            xy = sub[['center_x', 'center_y']].values
            straight, loops, n = straightness_and_loops(xy)
            row[f'{phase}_n'] = n
            row[f'{phase}_straightness'] = straight
            row[f'{phase}_loop_count'] = loops
        rows.append(row)

    df_out = pd.DataFrame(rows)

    dur = pd.read_csv('all_batch_durations.csv')
    df_out = df_out.merge(dur, on='batch_id', how='left')

    valid90 = set(pd.read_csv('all_valid_batches_dense2.csv')['batch_id'])
    df_out['in_90_valid'] = df_out['batch_id'].isin(valid90)

    art = pd.read_csv('trajectory_descriptions_dense2/trajectory_descriptions.csv')[['batch_id', 'artifact_flag']]
    df_out = df_out.merge(art, on='batch_id', how='left')

    out_path = 'trajectory_descriptions_dense2/trajectory_shape_metrics.csv'
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path}  ({len(df_out)} rows)")
    print(df_out[['batch_id', 'duration_sec', 'behavior', 'in_90_valid', 'artifact_flag',
                   'pre_n', 'pre_straightness', 'pre_loop_count',
                   'post_n', 'post_straightness', 'post_loop_count']].describe(include='all'))


if __name__ == '__main__':
    main()
