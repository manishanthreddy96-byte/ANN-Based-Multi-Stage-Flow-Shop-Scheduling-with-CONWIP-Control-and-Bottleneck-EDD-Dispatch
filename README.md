# ANN-Based-Multi-Stage-Flow-Shop-Scheduling-with-CONWIP-Control-and-Bottleneck-EDD-Dispatch
Below is a **clear explanation + documentation** for your total code (multi-stage ANN + CONWIP + bottleneck EDD + sensitivity + charts). This is written in **manager + engineer friendly** language.

---

# Project Name

**ANN-Based Multi-Stage Flow Shop Scheduling with CONWIP Control and Bottleneck EDD Dispatch**

---

# 1) What this code solves (in simple words)

A factory has **3 stages**:

1. **Prep**
2. **Process** (this is the slowest stage → bottleneck)
3. **Finish**

If too many jobs enter the line, waiting time increases and deliveries become late.

So this system:

* **Predicts how long each job will take at each stage** (using ANN)
* **Limits total jobs in the line** (CONWIP / WIP limit)
* Uses **EDD only at the bottleneck** to reduce late jobs
* Tests different WIP limits to find the **best sweet spot**

---

# 2) Inputs and outputs

## Inputs

No external file needed (it generates data automatically).

* Job features:

  * `product_type` (Standard/Custom/Premium)
  * `batch_size`
  * `priority`

## Outputs (files created)

1. `flowshop_sensitivity.csv`
   KPI results for different WIP limits
2. `flowshop_sensitivity.png`
   Graph: WIP vs Flow Time and Tardiness
3. `flowshop_gantt.png`
   Multi-stage Gantt chart for the chosen WIP limit

---

# 3) Code structure (what each part does)

## A) CONFIG section

```python
STAGES = ["Prep", "Process", "Finish"]
BOTTLENECK_STAGE = 1
SENS_LIMITS = [2,4,6,8,10,12,16]
GANTT_WIP = 10
```

Meaning:

* You define the 3 stages
* You declare **Process** as the bottleneck stage
* You test multiple WIP limits
* You pick one WIP limit to generate the Gantt chart

---

## B) `make_multi_stage_data()`

Creates realistic synthetic training data.

```python
prep_time  = fast
proc_time  = slow (bottleneck)
fin_time   = medium + variable
```

Why this is important:

* It simulates a real plant where **one stage limits throughput**
* This is how flow-shop systems behave in industry

---

## C) `train_multi_stage_ann()`

Trains one ANN model that predicts **3 outputs at the same time**:

* prep_time
* proc_time
* fin_time

Key steps:

1. Encodes product type using OneHotEncoder
2. Scales numeric features with StandardScaler
3. Trains ANN (MLPRegressor)

It also prints validation metrics:

* MAE
* RMSE

Why this matters:

* You prove the ANN is predicting realistically (not random)



## D) `simulate_flow_shop()` (core factory logic)

This is the **main engine**.

### Step 1: Predict stage times for all jobs

```python
p_prep, p_proc, p_fin
```

### Step 2: Set due date

```python
due_date = total_predicted_time * DUE_FACTOR
```

### Step 3: CONWIP release logic

```python
while backlog and len(active_wip_ids) < wip_limit:
    release job into stage 1 queue
```

Meaning:

* New jobs are released only if total WIP is below the limit
* This prevents overload

### Step 4: Stage processing + dispatch rule

For each stage, if machine is free:

* Prep queue → FIFO
* Process (bottleneck) → **EDD**
* Finish queue → FIFO

This is the key upgrade:

> Use EDD only where it matters most (bottleneck), so lateness reduces without creating chaos everywhere else.

### Step 5: Completion calculations

When job finishes last stage:

* total flow time
* tardiness
* remove from WIP

---

E) `compute_kpis()`

Computes key manufacturing KPIs:

* Jobs Completed = throughput
* Avg Flow Time = lead time
* Avg Tardiness = late minutes
* OTD% = on-time delivery

---

## F) Sensitivity analysis: `run_sensitivity()`

Runs the simulation multiple times for different WIP limits.

Example:

* WIP = 2
* WIP = 4
* …
* WIP = 16

This creates a table showing how:

* Flow time changes
* Tardiness changes
* OTD changes

This is **industry standard tuning**.

---

## G) Charts

### 1) `plot_sensitivity()`

Creates a curve to find the sweet spot.

Typical behavior:

* Low WIP → lower flow time, but lower throughput
* High WIP → high flow time, high tardiness
* Middle WIP → best balance

### 2) `plot_multi_gantt()`

Shows the first 12 jobs with stage bars:

* Prep row
* Process row
* Finish row

This is for:

* managers (visual)
* interview presentation
* GitHub screenshots

---

# 4) How to explain to an interviewer (simple line)

Say this:

> “I built a flow-shop model where an ANN predicts stage times, CONWIP controls WIP, and bottleneck EDD reduces lateness. Then I tune the WIP limit using sensitivity analysis.”

---

# 5) Why this is industry standard

This matches real manufacturing systems:

* ANN = predictive analytics
* CONWIP = Lean pull control
* Bottleneck dispatching = Theory of Constraints
* Sensitivity analysis = continuous improvement

---

# 6) What you can add to make it even stronger (optional)

Pick only one:

* Add FIFO vs Bottleneck-EDD comparison table
* Add stage utilization calculation (per stage)
* Add cost impact estimate (tardiness cost + WIP cost)

