# Spark Lab — Cloud Deployment on GCP (Dataproc) — CLI/Cloud Shell Workflow

This README documents the **exact process we used** to deploy and run our Spark analyses on **Google Cloud Platform (GCP)** using **Dataproc** from **Cloud Shell**, without relying on the Dataproc Console UI.

Our code is in GitHub (with the analysis in `src/project.ipynb` / exported to a PySpark script).  
The local dataset folder `spark-data/` is **not pushed**. In the cloud we read the dataset from **Google Cloud Storage (GCS)**, using only our assigned **sample parts 60–64**.

---

## 1) Prerequisites

- A GCP project with billing/credits enabled.
- Cloud Shell access (recommended).
- Dataproc cluster creation permissions.
- A Dataproc cluster already created OR we create one below.

---

## 2) Set environment variables

Run in **Cloud Shell**:

```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION=europe-west1

# Cluster name (the one we created)
export CLUSTER=spark-lab

# A globally-unique bucket name (example)
export BUCKET=uga-spark-lab-<unique-suffix>
```

## 3) Enable required APIs

```bash
gcloud services enable dataproc.googleapis.com storage.googleapis.com compute.googleapis.com
```

## 4) Create a GCS bucket for our job + outputs

```bash
gcloud storage buckets create gs://$BUCKET --location=$REGION

gcloud storage mkdir gs://$BUCKET/jobs
gcloud storage mkdir gs://$BUCKET/results
gcloud storage mkdir gs://$BUCKET/figures
```

## 5) (If needed) Fix permissions for Dataproc to access Cloud Storage

If cluster creation fails with Storage permission errors for the default compute service account, grant Storage permissions:

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/storage.admin"
```

**Note:** This is the fastest role for a lab setup. You can tighten it later.

## 6) Create the Dataproc cluster (with small disks to fit quota)

We used explicit boot disk sizes to avoid quota issues:

```bash
gcloud dataproc clusters create $CLUSTER \
  --region=$REGION \
  --master-machine-type=e2-standard-4 \
  --worker-machine-type=e2-standard-4 \
  --num-workers=4 \
  --master-boot-disk-size=100GB \
  --worker-boot-disk-size=100GB
```

Verify:

```bash
gcloud dataproc clusters list --region=$REGION
```

## 7) Get the code from GitHub in Cloud Shell

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```

## 8) Use only the sample data (parts 60–64)

### Option A — Read directly from the public dataset bucket (fastest)

We used GCS paths like:

- `gs://clusterdata-2011-2/task_events/part-00060-of-00500.csv.gz`
- `gs://clusterdata-2011-2/task_usage/part-00060-of-00500.csv.gz`

(and similarly 00061..00064)

### Option B (recommended for reproducibility) — Copy only parts 60–64 to our bucket

This avoids depending on the public bucket for every run:

```bash
gcloud storage mkdir gs://$BUCKET/data/task_events
gcloud storage mkdir gs://$BUCKET/data/task_usage
gcloud storage mkdir gs://$BUCKET/data/job_events
gcloud storage mkdir gs://$BUCKET/data/machine_events

# Copy sample parts 60..64
for i in $(seq -w 60 64); do
  gcloud storage cp gs://clusterdata-2011-2/task_events/part-000${i}-of-00500.csv.gz gs://$BUCKET/data/task_events/
  gcloud storage cp gs://clusterdata-2011-2/task_usage/part-000${i}-of-00500.csv.gz  gs://$BUCKET/data/task_usage/
  gcloud storage cp gs://clusterdata-2011-2/job_events/part-000${i}-of-00500.csv.gz   gs://$BUCKET/data/job_events/
done

# machine_events is a single file
gcloud storage cp gs://clusterdata-2011-2/machine_events/part-00000-of-00001.csv.gz gs://$BUCKET/data/machine_events/
```

Then in the code, we use:

```python
PATH_MACHINE_EVENTS = "gs://<bucket>/data/machine_events/*.csv.gz"
PATH_JOB_EVENTS     = "gs://<bucket>/data/job_events/*.csv.gz"
PATH_TASK_EVENTS    = "gs://<bucket>/data/task_events/*.csv.gz"
PATH_TASK_USAGE     = "gs://<bucket>/data/task_usage/*.csv.gz"
```

## 9) Run the notebook code on Dataproc (CLI-only)

Because Cloud Shell may not have Jupyter installed, we submit a PySpark script version of our notebook.

### 9.1 Export notebook to a script (done locally) and upload to GCS

On your local machine (recommended):

```bash
jupyter nbconvert --to script src/project.ipynb --output project_cloud.py
```

Upload it to GCS (from Cloud Shell or local terminal where gcloud is configured):

```bash
gcloud storage cp src/project_cloud.py gs://$BUCKET/jobs/project_cloud.py
```

If you already have `project_cloud.py`, just upload it.

### 9.2 Submit the job to Dataproc

We used environment variables to switch the input paths inside the script:

```bash
gcloud dataproc jobs submit pyspark gs://$BUCKET/jobs/project_cloud.py \
  --cluster=$CLUSTER --region=$REGION \
  --properties=spark.yarn.appMasterEnv.USE_GCS=1,spark.yarn.appMasterEnv.BUCKET_NAME=$BUCKET
```

Capture logs for the report:

```bash
gcloud dataproc jobs submit pyspark gs://$BUCKET/jobs/project_cloud.py \
  --cluster=$CLUSTER --region=$REGION \
  --properties=spark.yarn.appMasterEnv.USE_GCS=1,spark.yarn.appMasterEnv.BUCKET_NAME=$BUCKET \
  | tee run_${CLUSTER}.log

gcloud storage cp run_${CLUSTER}.log gs://$BUCKET/results/
```

## 10) Performance evaluation (2/4/8 workers)

The cloud task requires running in a distributed setup and doing a small performance evaluation.

We compared runtimes across different cluster sizes.

### 10.1 Create a 2-worker cluster

```bash
gcloud dataproc clusters create spark-lab-2w \
  --region=$REGION \
  --master-machine-type=e2-standard-4 \
  --worker-machine-type=e2-standard-4 \
  --num-workers=2 \
  --master-boot-disk-size=100GB \
  --worker-boot-disk-size=100GB
```

Run with timing:

```bash
time gcloud dataproc jobs submit pyspark gs://$BUCKET/jobs/project_cloud.py \
  --cluster=spark-lab-2w --region=$REGION \
  --properties=spark.yarn.appMasterEnv.USE_GCS=1,spark.yarn.appMasterEnv.BUCKET_NAME=$BUCKET \
  | tee run_spark-lab-2w.log

gcloud storage cp run_spark-lab-2w.log gs://$BUCKET/results/
```

### 10.2 Run on the 4-worker cluster (baseline)

```bash
time gcloud dataproc jobs submit pyspark gs://$BUCKET/jobs/project_cloud.py \
  --cluster=$CLUSTER --region=$REGION \
  --properties=spark.yarn.appMasterEnv.USE_GCS=1,spark.yarn.appMasterEnv.BUCKET_NAME=$BUCKET \
  | tee run_${CLUSTER}.log

gcloud storage cp run_${CLUSTER}.log gs://$BUCKET/results/
```

### 10.3 Create and run an 8-worker cluster (or 6 if quotas limit)

```bash
gcloud dataproc clusters create spark-lab-8w \
  --region=$REGION \
  --master-machine-type=e2-standard-4 \
  --worker-machine-type=e2-standard-4 \
  --num-workers=8 \
  --master-boot-disk-size=100GB \
  --worker-boot-disk-size=100GB
```

Then:

```bash
time gcloud dataproc jobs submit pyspark gs://$BUCKET/jobs/project_cloud.py \
  --cluster=spark-lab-8w --region=$REGION \
  --properties=spark.yarn.appMasterEnv.USE_GCS=1,spark.yarn.appMasterEnv.BUCKET_NAME=$BUCKET \
  | tee run_spark-lab-8w.log

gcloud storage cp run_spark-lab-8w.log gs://$BUCKET/results/
```

## 11) Download results / figures (optional)

```bash
mkdir -p out_results out_figures
gcloud storage cp gs://$BUCKET/results/* out_results/
gcloud storage cp gs://$BUCKET/figures/* out_figures/
```

## 12) Cleanup (important to avoid spending credits)

```bash
gcloud dataproc clusters delete $CLUSTER --region=$REGION
gcloud dataproc clusters delete spark-lab-2w --region=$REGION
gcloud dataproc clusters delete spark-lab-8w --region=$REGION
```

(Optional) remove the bucket:

```bash
gcloud storage rm -r gs://$BUCKET
```

## Notes / Dataset scope

All analyses were executed on the sample: parts 60 to 64 (task_events, task_usage, job_events), not on the full dataset.

We used GCS paths in the cloud instead of local `spark-data/`.

<!-- TO TEST FIRST : -->
<!-- ## 13) Running on the full dataset (optional experiment)

All the analyses in the report were conducted on our assigned sample (**parts 60–64**).  
In addition, we performed an **optional experiment** to run the same Spark pipeline on the **full dataset** directly from the public GCS bucket, by switching input paths to include **all parts**.

### Full dataset input paths (public bucket)

To run on the full dataset, we changed the paths in the cloud configuration to:

```python
PATH_MACHINE_EVENTS = "gs://clusterdata-2011-2/machine_events/part-00000-of-00001.csv.gz"
PATH_JOB_EVENTS     = "gs://clusterdata-2011-2/job_events/part-*.csv.gz"
PATH_TASK_EVENTS    = "gs://clusterdata-2011-2/task_events/part-*.csv.gz"
PATH_TASK_USAGE     = "gs://clusterdata-2011-2/task_usage/part-*.csv.gz"
```

This instructs Spark to read all partitions (e.g., `part-00000-of-00500.csv.gz` … `part-00499-of-00500.csv.gz`) for the large tables.

### Submitting the full-dataset run

We then submitted the job normally (same command as before), but with the script configured to use the full-dataset paths:

```bash
time gcloud dataproc jobs submit pyspark gs://$BUCKET/jobs/project_cloud.py \
  --cluster=$CLUSTER --region=$REGION \
  --properties=spark.yarn.appMasterEnv.USE_GCS=1,spark.yarn.appMasterEnv.BUCKET_NAME=$BUCKET \
  | tee run_${CLUSTER}_full.log

gcloud storage cp run_${CLUSTER}_full.log gs://$BUCKET/results/
```

This run is significantly heavier (especially `task_usage`), therefore it may require a larger cluster (more workers and/or larger disks).

To avoid driver memory issues, we ensured that the code does not call `collect()` on large RDDs and uses sampling for plots. -->
