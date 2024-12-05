#!/bin/bash 

# Configuration
APP_NAME="MySparkPythonApp"
MASTER="yarn"
DEPLOY_MODE="cluster"
SCRIPT_PATH="my_spark_script.py"
SUBSAMPLE="0.1"
NUM_EXECUTORS=5
EXECUTOR_CORES=2
EXECUTOR_MEMORY="40g"
DRIVER_MEMORY="16g"
LOGFILE_NAME="spark_job_$(date +%Y%m%d_%H%M%S).log"
S3_BUCKET="s3://ubs-cde/home/e2405193/bigdata/log_files"

# Run Spark-submit with nohup and log output to a file
nohup spark-submit \
    --name "$APP_NAME" \
    --master "$MASTER" \
    --deploy-mode "$DEPLOY_MODE" \
    --num-executors "$NUM_EXECUTORS" \
    --executor-cores "$EXECUTOR_CORES" \
    --executor-memory "$EXECUTOR_MEMORY" \
    --driver-memory "$DRIVER_MEMORY" \
    "$SCRIPT_PATH" \
    --subsample \
    | aws s3 cp - "$S3_BUCKET/$LOGFILE_NAME" &


