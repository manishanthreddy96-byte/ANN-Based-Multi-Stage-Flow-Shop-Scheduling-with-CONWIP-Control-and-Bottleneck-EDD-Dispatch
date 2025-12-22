import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# CONFIG
# ---------------------------
SEED = 42
STAGES = ["Prep", "Process", "Finish"]
TIME_HORIZON = 24 * 60  # minutes

# Bottleneck stage index (Process)
BOTTLENECK_STAGE = 1

# Sensitivity WIP limits
SENS_LIMITS = [2, 4, 6, 8, 10, 12, 16]

# Run a Gantt for this WIP
GANTT_WIP = 10

# Due-date tightness factor (bigger => easier due dates)
DUE_FACTOR = 2.5


# ---------------------------
# 1) Multi-Stage Synthetic Data
# ---------------------------
def make_multi_stage_data(n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "product_type": rng.choice(["Standard", "Custom", "Premium"], n),
        "batch_size": rng.integers(10, 100, n),
        "priority": rng.integers(1, 4, n)  # 1 low .. 3 high
    })

    # Prep fast, Process bottleneck, Finish variable
    df["prep_time"] = 5 + (df["batch_size"] * 0.1) + rng.normal(0, 1, n)
    df["proc_time"] = 20 + (df["batch_size"] * 0.5) + rng.normal(0, 5, n)
    df["fin_time"]  = 10 + (df["batch_size"] * 0.2) + rng.normal(0, 2, n)

    # Ensure no negatives
    for c in ["prep_time", "proc_time", "fin_time"]:
        df[c] = df[c].clip(lower=2)

    return df


# ---------------------------
# 2) Multi-Output ANN Training + Validation Metrics
# ---------------------------
def train_multi_stage_ann(df: pd.DataFrame):
    X = df[["product_type", "batch_size", "priority"]].copy()
    y = df[["prep_time", "proc_time", "fin_time"]].copy()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["product_type"]),
        ("num", StandardScaler(), ["batch_size", "priority"])
    ])

    ann = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=600,
        random_state=SEED
    )

    model = Pipeline([
        ("prep", preprocessor),
        ("ann", ann)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # aggregated metrics
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    print("\n=== Multi-Output ANN Validation ===")
    print(f"MAE  (avg across stages): {mae:.2f} min")
    print(f"RMSE (avg across stages): {rmse:.2f} min")

    # per-stage metrics for more insight
    stage_names = ["prep_time", "proc_time", "fin_time"]
    for i, sname in enumerate(stage_names):
        mae_i = mean_absolute_error(y_test.iloc[:, i], pred[:, i])
        rmse_i = mean_squared_error(y_test.iloc[:, i], pred[:, i]) ** 0.5
        print(f"  - {sname:9s} | MAE: {mae_i:.2f} min | RMSE: {rmse_i:.2f} min")

    return model


# ---------------------------
# 3) Flow Shop Simulation (CONWIP + Bottleneck EDD)
# ---------------------------
def simulate_flow_shop(jobs_pool: pd.DataFrame, model, wip_limit=10, time_horizon=TIME_HORIZON):
    jobs_pool = jobs_pool.copy()

    # Predict durations for all stages
    preds = model.predict(jobs_pool[["product_type", "batch_size", "priority"]])
    jobs_pool["p_prep"] = preds[:, 0]
    jobs_pool["p_proc"] = preds[:, 1]
    jobs_pool["p_fin"]  = preds[:, 2]

    # Due dates: based on predicted total time
    jobs_pool["due_date"] = (jobs_pool["p_prep"] + jobs_pool["p_proc"] + jobs_pool["p_fin"]) * DUE_FACTOR

    backlog = jobs_pool.to_dict("records")
    active_wip_ids = set()
    completed = []

    # Machine availability times (3 stages)
    m_avail = [0.0, 0.0, 0.0]

    # Queues per stage
    queues = [[], [], []]

    stage_key = {0: "p_prep", 1: "p_proc", 2: "p_fin"}

    current_time = 0.0

    while (backlog or any(queues) or active_wip_ids) and current_time < time_horizon:

        # 1) CONWIP release gate
        while backlog and len(active_wip_ids) < wip_limit:
            job = backlog.pop(0)
            job["enter_line_time"] = current_time
            queues[0].append(job)
            active_wip_ids.add(job["job_id"])

        progressed = False

        # 2) Dispatch/Process from Finish to Prep
        for i in range(2, -1, -1):

            if not queues[i]:
                continue

            # If machine is free at this stage
            if current_time >= m_avail[i]:

                # ---- DISPATCH RULE ----
                if i == BOTTLENECK_STAGE:
                    # Bottleneck uses EDD (earliest due date)
                    queues[i].sort(key=lambda j: (j["due_date"], j["job_id"]))
                else:
                    # Other stages FIFO (already in arrival order)
                    pass

                job = queues[i].pop(0)

                start = current_time
                duration = float(job[stage_key[i]])
                finish = start + duration

                m_avail[i] = finish
                job[f"s{i}_start"] = start
                job[f"s{i}_finish"] = finish

                # Move forward
                if i < 2:
                    queues[i + 1].append(job)
                else:
                    job["exit_line_time"] = finish
                    job["total_flow_time"] = finish - float(job["enter_line_time"])
                    job["tardiness"] = max(0.0, finish - float(job["due_date"]))
                    completed.append(job)
                    active_wip_ids.discard(job["job_id"])

                progressed = True

        # 3) Advance time safely
        future_events = [t for t in m_avail if t > current_time]
        if future_events:
            current_time = min(future_events)
        else:
            # no job in progress but perhaps backlog is empty and queues empty -> increment
            current_time += 1.0

    return pd.DataFrame(completed)


# ---------------------------
# 4) KPIs
# ---------------------------
def compute_kpis(out: pd.DataFrame) -> dict:
    if out.empty:
        return {
            "Jobs Completed": 0,
            "Avg Flow Time": 0.0,
            "Avg Tardiness": 0.0,
            "OTD %": 0.0,
        }

    ot = float((out["tardiness"] <= 1e-9).mean() * 100.0)
    return {
        "Jobs Completed": int(len(out)),
        "Avg Flow Time": float(out["total_flow_time"].mean()),
        "Avg Tardiness": float(out["tardiness"].mean()),
        "OTD %": ot,
    }


# ---------------------------
# 5) Sensitivity Analysis
# ---------------------------
def run_sensitivity(model, limits=SENS_LIMITS, n_jobs=200, seed=SEED):
    rows = []
    base_jobs = make_multi_stage_data(n_jobs, seed=seed)
    base_jobs["job_id"] = [f"Job_{i}" for i in range(len(base_jobs))]

    for w in limits:
        out = simulate_flow_shop(base_jobs, model, wip_limit=w)
        k = compute_kpis(out)
        k["WIP_Limit"] = w
        rows.append(k)

    return pd.DataFrame(rows).sort_values("WIP_Limit").reset_index(drop=True)


def plot_sensitivity(df_sens: pd.DataFrame, out_png="flowshop_sensitivity.png"):
    plt.figure(figsize=(10, 5))

    plt.plot(df_sens["WIP_Limit"], df_sens["Avg Flow Time"], marker="o", label="Avg Flow Time")
    plt.plot(df_sens["WIP_Limit"], df_sens["Avg Tardiness"], marker="x", linestyle="--", label="Avg Tardiness")

    plt.xlabel("CONWIP / WIP Limit (jobs)")
    plt.ylabel("Minutes")
    plt.title("Sensitivity: WIP Limit vs Flow Time & Tardiness (Bottleneck EDD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# 6) Multi-Machine Gantt
# ---------------------------
def plot_multi_gantt(df: pd.DataFrame, out_png="flowshop_gantt.png", num_jobs=12):
    if df.empty:
        print("[WARN] No completed jobs for Gantt.")
        return

    dfp = df.head(num_jobs).copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Prep, Process, Finish
    y_positions = list(range(len(dfp)))

    for row_idx, (_, job) in enumerate(dfp.iterrows()):
        for s in range(3):
            start = float(job.get(f"s{s}_start", np.nan))
            fin = float(job.get(f"s{s}_finish", np.nan))
            if np.isnan(start) or np.isnan(fin):
                continue
            width = fin - start
            ax.barh(row_idx, width, left=start, height=0.6, color=colors[s], edgecolor="black", alpha=0.9)
            ax.text(start + width / 2, row_idx, STAGES[s], ha="center", va="center", fontsize=7, color="white")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(dfp["job_id"].tolist())
    ax.set_xlabel("Time (Minutes)")
    ax.set_title(f"Multi-Stage Flow Shop Gantt (CONWIP Limit={GANTT_WIP}, Bottleneck=EDD)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[i], label=STAGES[i]) for i in range(3)]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("[1/4] Generating training data...")
    raw_data = make_multi_stage_data(n=4000)

    print("[2/4] Training multi-output ANN...")
    ann_model = train_multi_stage_ann(raw_data)

    print("[3/4] Running sensitivity analysis...")
    sens = run_sensitivity(ann_model, limits=SENS_LIMITS, n_jobs=220)
    sens.to_csv("flowshop_sensitivity.csv", index=False)
    plot_sensitivity(sens, out_png="flowshop_sensitivity.png")

    print("\n=== SENSITIVITY RESULTS (Bottleneck EDD) ===")
    print(sens.to_string(index=False))

    print("[4/4] Running one simulation for Gantt...")
    test_jobs = make_multi_stage_data(n=120, seed=SEED + 7)
    test_jobs["job_id"] = [f"Job_{i}" for i in range(len(test_jobs))]

    results = simulate_flow_shop(test_jobs, ann_model, wip_limit=GANTT_WIP)

    print("\n=== SYSTEM PERFORMANCE (Gantt Run) ===")
    perf = compute_kpis(results)
    for k, v in perf.items():
        if isinstance(v, float):
            print(f"{k:<15}: {v:.2f}")
        else:
            print(f"{k:<15}: {v}")

    plot_multi_gantt(results, out_png="flowshop_gantt.png", num_jobs=12)

    print("\n[OK] Saved outputs:")
    print(" - flowshop_sensitivity.csv")
    print(" - flowshop_sensitivity.png")
    print(" - flowshop_gantt.png")