
'''
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# ---------------------------------------------------------------------------
# 1 – Import the environment class you already defined in rl_agent.py
# ---------------------------------------------------------------------------
from rl_agent import ThreeSensorTimeEnv

SENSOR_COSTS = [10, 4, 1]  # ECG, PPG, Temp  (mA for a 5‑s window)

# ---------------------------------------------------------------------------
# 2 – Load Q‑table utility ----------------------------------------------------
State = tuple  # any length >=5
QTable = Dict[Tuple[State, int], float]


def load_q_table(csv_path: Path) -> QTable:
    """
    Read qtable CSV produced by rl_agent.py.

    Accepts either 7-column (without prev_arr) or 8-column (with prev_arr) format.
    """
    out: QTable = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)            # skip header
        for row in reader:
            if len(row) == 7:
                b, t, arr, bp, fev, act, qv = row
                state = (int(b), int(t), int(arr), int(bp), int(fev))
            elif len(row) == 8:
                b, t, arr, bp, fev, prev_arr, act, qv = row
                state = (
                    int(b),
                    int(t),
                    int(arr),
                    int(bp),
                    int(fev),
                    int(prev_arr),
                )
            else:
                raise ValueError(f"Unexpected row length {len(row)} in Q-table")

            out[(state, int(act))] = float(qv)
    return out

# ---------------------------------------------------------------------------
# 3 – Policy helpers ---------------------------------------------------------

def greedy_rl(Q: QTable, state: State) -> int:
    qvals = [Q.get((state, a), 0.0) for a in range(8)]
    return int(np.argmax(qvals))


def always_on(_: State) -> int:
    return 0b111  # ECG, PPG, Temp on


def periodic_ecg(state: State) -> int:
    """Turn ECG ON every 30 s (6 time‑steps at 5 s cadence)."""
    _, t, *_ = state
    return 0b100 if (t % 6) == 0 else 0b000

# ---------------------------------------------------------------------------
# 4 – Synthetic anomaly scenario --------------------------------------------

def synthetic_scenario(T: int) -> List[dict]:
    rng = np.random.default_rng(0)
    return [
        {
            "arr_flag": rng.choice([0, 1], p=[0.9, 0.1]),
            "bp_flag": rng.choice([0, 1], p=[0.7, 0.3]),
            "fever_flag": rng.choice([0, 1], p=[0.9, 0.1]),
        }
        for _ in range(T)
    ]

# ---------------------------------------------------------------------------
# 5 – Episode runner ---------------------------------------------------------

def run_episode(env: ThreeSensorTimeEnv, policy_fn, label: str) -> dict:
    """Execute one full episode and collect detection/energy stats."""
    state = env.reset()
    det_hits = det_total = energy_cost = 0

    while True:
        action = policy_fn(state)
        next_state, _, done, _ = env.step(action)

        # --- energy bookkeeping -------------------------------------------
        ecg_on = (action >> 2) & 1
        ppg_on = (action >> 1) & 1
        tmp_on = action & 1
        energy_cost += (
            SENSOR_COSTS[0] * ecg_on
            + SENSOR_COSTS[1] * ppg_on
            + SENSOR_COSTS[2] * tmp_on
        )

        # --- anomaly detection bookkeeping --------------------------------
        for flag, on in zip(
            [env.arr_flag, env.bp_flag, env.fever_flag],
            [ecg_on, ppg_on, tmp_on],
        ):
            if flag:
                det_total += 1
                if on:
                    det_hits += 1

        if done:
            break
        state = next_state

    detection_rate = det_hits / det_total * 100 if det_total else 0.0
    mAh = energy_cost * 5 / 3600          # convert mA·5 s → mAh
    return dict(label=label, detection=detection_rate, mAh=mAh)

# ---------------------------------------------------------------------------
# 6 – Main: run three policies ----------------------------------------------
if __name__ == "__main__":
    STEPS = 12_000                       # ≈16 h at 5‑s cadence
    scenario = synthetic_scenario(STEPS)

    # Build environment factory so battery resets each run
    def make_env():
        return ThreeSensorTimeEnv(
            scenario,
            sensor_costs=SENSOR_COSTS,
            alpha=4.0,
            beta=0.008,
            max_battery=400_000,
            max_time_steps=STEPS,
        )

    Q = load_q_table(Path("qtable_sensors_time_ff.csv"))

    policies = [
        (lambda s: greedy_rl(Q, s), "RL"),
        (always_on, "Always‑on"),
        (periodic_ecg, "Periodic‑5/30"),
    ]

    results = []
    for fn, name in policies:
        env = make_env()
        results.append(run_episode(env, fn, name))

    print(f"\nDetection vs. Energy (synthetic {STEPS}‑step run)")
    print("Label           Det. (%)   Energy (mAh)")
    for res in results:
        print(f"{res['label']:<15s} {res['detection']:7.1f}      {res['mAh']:8.2f}")'''
