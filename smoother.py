
"""
smoother.py
-----------
Post-processing functions for fencing tip tracking.
"""

import math
import numpy as np
from typing import List, Tuple


def fill_missing_tips_farthest(tips_history, expect_count=2):
    T = len(tips_history)
    tracks = [np.full((T, 2), np.nan, dtype=float) for _ in range(expect_count)]
    is_missing = []

    def get_tips_at(i):
        item = tips_history[i]
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (list, tuple)):
            return item[1]
        return item

    def dist(a, b):
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    for t in range(T):
        tips = get_tips_at(t) or []
        m = len(tips)

        prev, prev_known = [], []
        for i in range(expect_count):
            if t > 0 and not np.isnan(tracks[i][t - 1, 0]):
                prev.append(tracks[i][t - 1])
                prev_known.append(True)
            else:
                prev.append(np.array([np.nan, np.nan], dtype=float))
                prev_known.append(False)

        if m >= expect_count:
            if all(prev_known) and expect_count == 2:
                c00 = dist(tips[0], prev[0]) + dist(tips[1], prev[1])
                c01 = dist(tips[0], prev[1]) + dist(tips[1], prev[0])
                perm = (0, 1) if c00 <= c01 else (1, 0)
                for i in range(expect_count):
                    tracks[i][t] = (float(tips[perm[i]][0]), float(tips[perm[i]][1]))
            else:
                for i in range(expect_count):
                    tracks[i][t] = (float(tips[i][0]), float(tips[i][1]))
            is_missing.append([0, 0])

        elif m == 1:
            d = (float(tips[0][0]), float(tips[0][1]))
            temp_missing = [0, 0]
            if all(prev_known) and expect_count == 2:
                d0, d1 = dist(d, prev[0]), dist(d, prev[1])
                near_idx = 0 if d0 <= d1 else 1
                far_idx = 1 - near_idx
                tracks[near_idx][t] = d
                tracks[far_idx][t] = (float(prev[far_idx][0]), float(prev[far_idx][1]))
                temp_missing[near_idx] = 0
                temp_missing[far_idx] = 1
            else:
                tracks[0][t] = d
                temp_missing[1] = 1
            is_missing.append(temp_missing)

        else:
            is_missing.append([1, 1])

    for i in range(expect_count):
        for t in range(1, T):
            if np.isnan(tracks[i][t, 0]) and not np.isnan(tracks[i][t - 1, 0]):
                tracks[i][t] = tracks[i][t - 1]
        idxs = np.where(~np.isnan(tracks[i][:, 0]))[0]
        if len(idxs) > 0:
            t0 = idxs[0]
            for t in range(0, t0):
                tracks[i][t] = tracks[i][t0]

    return tracks, is_missing


def fill_tips_midpoint(tips_history, missing_2col):
    tips = np.asarray([[a, b] for a, b in tips_history], dtype=float)
    T = tips.shape[0]
    miss = np.asarray(missing_2col, dtype=bool)
    tips_est = tips.copy()

    for tip_idx in (0, 1):
        valid = (~miss[:, tip_idx]) & np.all(np.isfinite(tips[:, tip_idx, :]), axis=1)

        prev_idx = np.full(T, -1, dtype=int)
        last = -1
        for t in range(T):
            if valid[t]: last = t
            prev_idx[t] = last

        next_idx = np.full(T, -1, dtype=int)
        nxt = -1
        for t in range(T - 1, -1, -1):
            if valid[t]: nxt = t
            next_idx[t] = nxt

        need_fill = miss[:, tip_idx] | (~np.all(np.isfinite(tips[:, tip_idx, :]), axis=1))

        for t in np.where(need_fill)[0]:
            pi, ni = prev_idx[t], next_idx[t]
            if pi >= 0 and ni >= 0 and pi != ni:
                tips_est[t, tip_idx, :] = 0.5 * (tips[pi, tip_idx, :] + tips[ni, tip_idx, :])
            elif pi >= 0:
                tips_est[t, tip_idx, :] = tips[pi, tip_idx, :]
            elif ni >= 0:
                tips_est[t, tip_idx, :] = tips[ni, tip_idx, :]
            else:
                tips_est[t, tip_idx, :] = np.array([np.nan, np.nan], dtype=float)

    return tips_est
