from pyspark.sql import SparkSession
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F
import time
from pyspark.storagelevel import StorageLevel
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import os

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# GCS_MODE:
#   - "local"        -> reads ../spark-data/...
#   - "bucket_sample"-> reads gs://<BUCKET>/data/... (your copied parts 60–64)
#   - "public_full"  -> reads gs://clusterdata-2011-2/.../part-*.csv.gz (whole dataset)
GCS_MODE = os.environ.get("GCS_MODE", "local").lower()
BUCKET = os.environ.get("uga-spark-lab-212121", "").strip()

if GCS_MODE in ("bucket_sample",) and not BUCKET:
    raise ValueError("BUCKET_NAME must be set when GCS_MODE=bucket_sample")

if GCS_MODE == "bucket_sample":
    # Your bucket contains only parts 60–64 (because you copied them there)
    PATH_MACHINE_EVENTS = f"gs://{BUCKET}/data/machine_events/*.csv.gz"
    PATH_JOB_EVENTS     = f"gs://{BUCKET}/data/job_events/*.csv.gz"
    PATH_TASK_EVENTS    = f"gs://{BUCKET}/data/task_events/*.csv.gz"
    PATH_TASK_USAGE     = f"gs://{BUCKET}/data/task_usage/*.csv.gz"

elif GCS_MODE == "public_full":
    # Full dataset from the public bucket (all parts)
    PATH_MACHINE_EVENTS = "gs://clusterdata-2011-2/machine_events/part-00000-of-00001.csv.gz"
    PATH_JOB_EVENTS     = "gs://clusterdata-2011-2/job_events/part-*.csv.gz"
    PATH_TASK_EVENTS    = "gs://clusterdata-2011-2/task_events/part-*.csv.gz"
    PATH_TASK_USAGE     = "gs://clusterdata-2011-2/task_usage/part-*.csv.gz"

else:
    # local
    PATH_MACHINE_EVENTS = "../spark-data/machine_events/*.csv.gz"
    PATH_JOB_EVENTS     = "../spark-data/job_events/*.csv.gz"
    PATH_TASK_EVENTS    = "../spark-data/task_events/*.csv.gz"
    PATH_TASK_USAGE     = "../spark-data/task_usage/*.csv.gz"

# ----------------------------
# Parsing utilities
# ----------------------------
def to_int(x):
    return int(x) if x not in (None, "", "NULL") else None

def to_long(x):
    return int(x) if x not in (None, "", "NULL") else None

def to_float(x):
    return float(x) if x not in (None, "", "NULL") else None

# ---
# ## Question 8 — *Are the tasks that request the more resources the one that consume the more resources?*
# 
# Information about the request of the resources can be found in *task_events*, while the real consumption of the resources is recorded in *task_usage*. <br>
# The first analysis is based on the comparison of requested and used **RAM**: <br>
# - From *task_events* are extracted the task IDs (job_id, task_idx) and the corresponding requested RAM, but removing duplicates (in case multiple events are recorded per task).
# - From *task_usage* are extracted the task IDs and, to simplify, the mean of the recorded usages of RAM.
# 
# To answer this question, there are two indicators:
# - a visualisation using a **heatmap**, by categorising the values into 5 bins based on the quantiles (in order to have useful quantities for each bin)
# - a **correlation** indicator that is more exact based on the raw values (real values before binning)
# 
# As it is shown by both approaches, the tasks requiring the most resources aquire the most resourses. That is explained by the heatmap colors concentrated in the diagonal and the correlation value being strong.<br>
# 
# The same analysis can be done for the other resources such as CPU and disk usage.
# 
# 
# 
# 

# In[40]:


# Loading the dataset
task_usage_rdd = sc.textFile(PATH_TASK_USAGE)
task_events_rdd = sc.textFile(PATH_TASK_EVENTS)

task_usage_rdd = (
    task_usage_rdd
    .map(lambda line: line.split(","))      # transforming strings into fields
)

# Exploring the dataset
CPU_values_events_table = (
    task_events_rdd
    .map(lambda line: line.split(","))
    .map(lambda x : float(x[9]))
    .distinct()
)
print("\n" + "="*80)
print(f"Number of records in task_usage: {task_usage_rdd.count()}")
print(f"Total of CPU_values_events_table : {CPU_values_events_table.count()}")
print(f"Min of CPU_values_events_table : {CPU_values_events_table.min()}")
print(f"Max of CPU_values_events_table : {CPU_values_events_table.max()}")


# In[41]:


task_events_reqRAM = (          # comes from task_events and concentrates on reqRAM
    task_events_rdd
    .map(lambda line: line.split(","))
    .map(lambda x: ((x[2], x[3]), float(x[10])))  # ( (job_id, task_idx), reqRAM )
    .distinct()                 # removing duplicates     
)

task_usage_usedRAM = (      # comes from task_usage and concentrates on usedRAM
    task_usage_rdd
    .map(lambda x: ((x[2], x[3]), float(x[6])))  # ( (job_id, task_idx), used RAM )
    .groupByKey()
    .mapValues(lambda x: np.mean(list(x)))  # average across all measurements
)

# Quantiles for req RAM
req_distribution = task_events_reqRAM.map(lambda x: x[1]).collect()
req_quantiles = np.quantile(req_distribution, [0.2, 0.4, 0.6, 0.8, 1.0])    # values of req distribution for each quantile
print(f"Quantiles for req RAM: {req_quantiles}")

# Quantiles for used RAM
used_distribution = task_usage_usedRAM.map(lambda x: x[1]).collect()
used_quantiles = np.quantile(used_distribution, [0.2, 0.4, 0.6, 0.8, 1.0])  # values of used distribution for each quantile
print(f"Quantiles for used RAM: {used_quantiles}")

# Binning functions to categorise values into 5 bins 
# each bin corresponds to the quantile
def reqRAM_value_to_range(x):
    if float(x) < req_quantiles[0]:   return 0
    elif float(x) < req_quantiles[1]: return 1
    elif float(x) < req_quantiles[2]: return 2
    elif float(x) < req_quantiles[3]: return 3
    else: return 4
def usedRAM_value_to_range(x):
    if float(x) < used_quantiles[0]:   return 0
    elif float(x) < used_quantiles[1]: return 1
    elif float(x) < used_quantiles[2]: return 2
    elif float(x) < used_quantiles[3]: return 3
    else: return 4

# Apply binning
task_events_reqRAM_binned = (
    task_events_reqRAM
    .map(lambda x: (x[0], reqRAM_value_to_range(x[1])))  # ((job_id, task_idx), req_bin)
)
task_usage_usedRAM_binned = (
    task_usage_usedRAM
    .map(lambda x: (x[0], usedRAM_value_to_range(x[1])))  # ((job_id, task_idx), used_bin)
)

# Joining req and used 
joined_req_used_RAM = task_events_reqRAM_binned.join(task_usage_usedRAM_binned)     # ((job_id, task_idx), (req_bin, used_bin))

# Counting for every combination between req and used bins
req_used_RAM_counts = (
    joined_req_used_RAM
    .map(lambda x: ((x[1][0], x[1][1]), 1))
    .reduceByKey(lambda a, b: a + b) 
)
data = req_used_RAM_counts.collect()


######################################################
#################### VISUALISATION ###################
######################################################

df = pd.DataFrame(
    [(x1, x2, count) for ((x1, x2), count) in data],
    columns=["req_bin", "used_bin", "count"]
)

heatmap_matrix = df.pivot(index="used_bin", columns="req_bin", values="count").fillna(0)

# Bin labels
req_labels = [
    f"Q1: <{req_quantiles[0]:.4f}",
    f"Q2: <{req_quantiles[1]:.4f}",
    f"Q3: <{req_quantiles[2]:.4f}",
    f"Q4: <{req_quantiles[3]:.4f}",
    f"Q5: <{req_quantiles[4]:.4f}"
]
used_labels = [
    f"Q1: <{used_quantiles[0]:.4f}",
    f"Q2: <{used_quantiles[1]:.4f}",
    f"Q3: <{used_quantiles[2]:.4f}",
    f"Q4: <{used_quantiles[3]:.4f}",
    f"Q5: <{used_quantiles[4]:.4f}"
]

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    heatmap_matrix, 
    annot=True,
    fmt='g',
    cmap='YlOrRd',
    cbar_kws={'label': 'Number of Tasks'},
    linewidths=0.5,
    linecolor='gray',
    ax=ax
)

ax.invert_yaxis()

ax.set_xlabel('Req RAM Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Used RAM Category', fontsize=12, fontweight='bold')
ax.set_title(
    'RAM: Req vs Used Resources\n(Binned by Quintiles)',
    fontsize=14,
    fontweight='bold',
    pad=20
)

ax.set_xticklabels(req_labels, rotation=45, ha='right')
ax.set_yticklabels(used_labels, rotation=0)

out = "heatmap.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print("Saved figure:", out)


######################################################
#################### STATISTICS ######################
######################################################

print("\n" + "="*80)
print("STATISTICS")

total_tasks = heatmap_matrix.sum().sum()
print(f"Total tasks: {total_tasks:.0f}")

# Calculate correlation on raw values (not binned)
req_used_values = (
    task_events_reqRAM
    .join(task_usage_usedRAM)
    .map(lambda x: (x[1][0], x[1][1]))  # (req, used)
    .filter(lambda x: x[0] > 0 and x[1] > 0)  # filter zeros
    .collect()
)
req_vals = [x[0] for x in req_used_values]
used_vals = [x[1] for x in req_used_values]
correlation, p_value = pearsonr(req_vals, used_vals)
print(f"Pearson Correlation: {correlation:.4f}")
print(f"Interpretation: ", end="")
if correlation > 0.7:
    print("STRONG positive correlation")
elif correlation > 0.3:
    print("MODERATE positive correlation")
else:
    print("WEAK correlation")


# The average locality ratio is 0.994, which is very close to 1. This indicates that, for most jobs, the number of machines used is almost equal to the number of tasks. In other words, tasks belonging to the same job are generally scheduled on different machines, and task co-location is rare.
# 
# This observation shows that the scheduler favors a distributed execution strategy, prioritizing parallelism and load balancing over strict task locality. By spreading tasks across many machines, the cluster can maximize throughput and reduce job execution time, especially for jobs with a large number of independent tasks.
# 
# However, this strategy also has drawbacks. Distributing tasks across multiple machines reduces data locality and may increase network communication and synchronization overhead. Despite these costs, the observed behavior is consistent with the design of large-scale cloud infrastructures, where resource utilization and fairness across users are more critical than keeping tasks from the same job on a single machine.
# 
# In general, tasks from the same job do not run on the same machine. The observed scheduling strategy prioritizes throughput and fairness over locality, which is consistent with the design of large-scale cloud infrastructures.

# ---

# ## Q9 : Correlation between CPU peaks and task evictions
# 
# **Method / definitions.**
# - Define machine-window key: `(machine_id, window_start)` where `window_start` is the task_usage start time.
# - Compute:
#   - `cpu_sum(machine, window)`: sum of CPU usage across tasks on that machine during the window.
#   - `evict_count(machine, window)`: number of eviction events mapped into the same 5-min bucket.
# - Define **peak windows** as top 5% windows by `cpu_sum` (95th percentile threshold).
# - Compare eviction activity in peak vs non-peak windows and compute Pearson correlation between `cpu_sum` and `evict_count`.


EVICT = 2
WINDOW_US = 5 * 60 * 1_000_000  # 5 minutes in microseconds

# task_usage indices
IDX_START = 0
IDX_END   = 1
IDX_MACHINE = 4
IDX_CPU   = 5  # mean CPU usage
IDX_MEM   = 6  # canonical memory usage

def floor_to_window(ts_us, window_us=WINDOW_US):
    return ts_us - (ts_us % window_us)

# ----------------------------
# Parse task_usage
# ----------------------------
def parse_task_usage(line):
    p = line.split(",")
    if len(p) <= max(IDX_MEM, IDX_CPU, IDX_MACHINE, IDX_END):
        return None

    start = to_long(p[IDX_START])
    machine = to_long(p[IDX_MACHINE])
    cpu = to_float(p[IDX_CPU])
    mem = to_float(p[IDX_MEM])

    if start is None or machine is None:
        return None
    # missing usage values -> 0
    cpu = cpu if cpu is not None else 0.0
    mem = mem if mem is not None else 0.0

    wstart = floor_to_window(start)
    return ((machine, wstart), (cpu, mem, 1))

task_usage = sc.textFile(PATH_TASK_USAGE).map(parse_task_usage).filter(lambda x: x is not None)

# Aggregate machine usage per window
usage_by_machine_window = (
    task_usage
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]))
    .cache()
)

print("Usage machine-windows:", usage_by_machine_window.count())

# ----------------------------
# Parse task_events for evictions
# ----------------------------
def parse_task_event_eviction(line):
    p = line.split(",")
    if len(p) < 6:
        return None

    ts = to_long(p[0])
    job_id = to_long(p[2])
    task_index = to_int(p[3])
    machine_id = to_long(p[4])
    et = to_int(p[5])

    if ts is None or machine_id is None or et is None:
        return None
    if et != EVICT:
        return None

    wstart = floor_to_window(ts)
    return ((machine_id, wstart), 1)

evictions_by_machine_window = (
    sc.textFile(PATH_TASK_EVENTS)
    .map(parse_task_event_eviction)
    .filter(lambda x: x is not None)
    .reduceByKey(lambda a,b: a+b)
    .cache()
)

print("Eviction machine-windows:", evictions_by_machine_window.count())

# ----------------------------
# Join usage with evictions
# ----------------------------
joined = (usage_by_machine_window
    .leftOuterJoin(evictions_by_machine_window)
    .mapValues(lambda v: (v[0], v[1] if v[1] is not None else 0))
    .cache()
)

print("Joined windows:", joined.count())

# Flatten for analysis
cpu_evict = joined.map(lambda x: (x[1][0][0], x[1][1])).cache()

# ----------------------------
# Peak threshold using sampling
# ----------------------------
sample = cpu_evict.map(lambda x: x[0]).sample(False, 0.02, seed=42).collect()  # 2% sample
sample = [v for v in sample if v is not None]
cpu_p95 = float(np.quantile(sample, 0.95)) if sample else 0.0

print("Approx CPU 95th percentile threshold (cpu_sum):", cpu_p95)

# ----------------------------
# Compare eviction behavior: peak vs non-peak
# ----------------------------
peak = cpu_evict.filter(lambda x: x[0] >= cpu_p95)
nonpeak = cpu_evict.filter(lambda x: x[0] < cpu_p95)

peak_n = peak.count()
nonpeak_n = nonpeak.count()

peak_evicted_windows = peak.filter(lambda x: x[1] > 0).count()
nonpeak_evicted_windows = nonpeak.filter(lambda x: x[1] > 0).count()

peak_rate = (peak_evicted_windows / peak_n) if peak_n else 0.0
nonpeak_rate = (nonpeak_evicted_windows / nonpeak_n) if nonpeak_n else 0.0

peak_avg_evict = peak.map(lambda x: x[1]).mean() if peak_n else 0.0
nonpeak_avg_evict = nonpeak.map(lambda x: x[1]).mean() if nonpeak_n else 0.0

print("\n--- Peak vs Non-Peak ---")
print("Peak windows:", peak_n, " Non-peak windows:", nonpeak_n)
print("P(eviction>0 | peak)   =", peak_rate)
print("P(eviction>0 | nonpeak)=", nonpeak_rate)
print("Avg evictions/window (peak)   =", peak_avg_evict)
print("Avg evictions/window (nonpeak)=", nonpeak_avg_evict)

# ----------------------------
# Pearson correlation between cpu_sum and eviction_count
# ----------------------------
def seq_op(acc, xy):
    n, sx, sy, sxx, syy, sxy = acc
    x, y = xy
    n += 1
    sx += x
    sy += y
    sxx += x*x
    syy += y*y
    sxy += x*y
    return (n, sx, sy, sxx, syy, sxy)

def comb_op(a, b):
    return tuple(a[i] + b[i] for i in range(6))

n, sx, sy, sxx, syy, sxy = cpu_evict.aggregate((0,0.0,0.0,0.0,0.0,0.0), seq_op, comb_op)

num = n * sxy - sx * sy
den = ((n * sxx - sx*sx) ** 0.5) * ((n * syy - sy*sy) ** 0.5)
pearson = (num / den) if den != 0 else 0.0

print("\nPearson correlation (cpu_sum vs eviction_count):", pearson)

# ----------------------------
# Plot: average evictions by CPU usage decile (optional)
# ----------------------------
# Build deciles from a sample and bucket windows accordingly
dec_sample = cpu_evict.map(lambda x: x[0]).sample(False, 0.02, seed=7).collect()
dec_sample = [v for v in dec_sample if v is not None]
qs = [float(np.quantile(dec_sample, q)) for q in np.linspace(0, 1, 11)] if dec_sample else [0.0]*11

def decile_bucket(x):
    # returns 0..9
    for i in range(10):
        if x <= qs[i+1]:
            return i
    return 9

binned = cpu_evict.map(lambda x: (decile_bucket(x[0]), (x[1], 1)))
dec_stats = (
    binned.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))
          .mapValues(lambda s: s[0]/s[1] if s[1] else 0.0)
          .collect()
)
dec_stats = sorted(dec_stats, key=lambda x: x[0])

xs = [d for d,_ in dec_stats]
ys = [v for _,v in dec_stats]

plt.figure(figsize=(8,4))
plt.bar([str(x) for x in xs], ys)
plt.xlabel("CPU usage decile (0=low, 9=high)")
plt.ylabel("Avg evictions per window")
plt.title("Q9: Evictions vs CPU usage level (deciles)")

out = "q9_evictions_vs_cpu_deciles.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print("Saved figure:", out)


# **Results (subset).**
# - Usage windows: **2,099,250**
# - Windows with ≥1 eviction: **26,312**
# - Joined windows: **2,101,869**
# - CPU 95th percentile threshold: `cpu_sum ≈ 0.4095`
# 
# Peak vs non-peak:
# - Peak windows: **103,781**
# - Non-peak windows: **1,998,088**
# - `P(eviction>0 | peak)` = **0.02255** (2.255%)
# - `P(eviction>0 | non-peak)` = **0.011997** (1.200%)
# - Avg evictions/window: peak **0.03642**, non-peak **0.01957**
# - Pearson correlation: **r = 0.0556**
# 
# **Interpretation.**
# - Evictions are about **~1.9× more likely** during peak CPU windows than during non-peak windows.
# - The Pearson correlation is small but positive, meaning CPU load alone is not a strong linear predictor.
# - Overall, results suggest that resource pressure increases eviction probability, but evictions also depend on other factors (priority, scheduling class, memory pressure, policies).