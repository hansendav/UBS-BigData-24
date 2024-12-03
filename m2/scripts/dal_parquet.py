# Import libraries 
from pyspark.sql import SparkSession 
from pyspark.sql.types import * # import all datatypes
import pyarrow.fs as fs
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as f
import rasterio 
import re 
import numpy as np
import argparse
import time

# -----------------------------------------------------------------------------
# ### Define decorator functions
# -----------------------------------------------------------------------------

def log_runtime(task_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_time_formatted = time.strftime("%H:%M:%S", time.localtime(start_time))
            print(f"{task_name} started at {start_time_formatted}")
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_time_formatted = time.strftime("%H:%M:%S", time.localtime(end_time))
            print(f"{task_name} finished at {end_time_formatted}")
            
            runtime = end_time - start_time
            hours, rem = divmod(runtime, 3600)
            minutes, seconds = divmod(rem, 60)
            runtime_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            print(f"Runtime of task {task_name}: {runtime_formatted}")
            
            return result 
        return wrapper
    return  decorator

def print_start_finish(whatstarted_message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"{whatstarted_message} started")
            result = func(*args, **kwargs)
            print(f"{whatstarted_message} finished")
            return result 
        return wrapper 
    return decorator

# -----------------------------------------------------------------------------
# ### Define functions
# -----------------------------------------------------------------------------
def prepare_cu_metadata(metadata):
    metadata = metadata \
    .withColumn('s1_path', f.split(f.col('s1_path'), 's3://').getItem(1))\
    .withColumn('s2_path', f.split(f.col('s2_path'), 's3://').getItem(1))\
    .withColumn('label_path', f.split(f.col('label_path'), 's3://').getItem(1))\
    .withColumn(
        'paths_array',
        f.array(
            f.col('s1_path'),
            f.col('s2_path'),
            f.col('label_path')
        )
    )\
    .select('paths_array', 'split')

    return metadata


def get_band_paths(patch_path, is_s2=False):
    """
    Extracts image band paths from a given directory path. 
    ---
    Example: Input: path to S2 directory holding all band tifs 
    Output: list of paths to all bands
    """

    # Uses pyarrow here to get list of files in the S3 directories
    filesystem = fs.S3FileSystem()

    if is_s2 == True:
        files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
        file_paths = ['s3://' + file.path for file in files_info if file.is_file and re.search(r'_B(0[2348]).tif$', file.path)]
    else: 
        files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
        file_paths = ['s3://' + file.path for file in files_info if file.is_file] 
    
    return file_paths

def read_band(band_path): 
        with rasterio.open(band_path) as src:
            band = src.read()
        return band

def read_bands(band_paths):
    bands = [read_band(band_path) for band_path in band_paths]
    return bands

def get_paths_from_meta(self, patch_path_array):
    rows = patch_path_array
    s1_path = rows[0]
    s2_path = rows[1]
    label_path = rows[2]

    return s1_path, s2_path, label_path


def stack_image_array(patch_path_array):

    s1_path, s2_path, label_path = get_paths_from_meta(patch_path_array)

    s2_band_paths = get_band_paths(s2_path, is_s2=True)
    s1_band_paths = get_band_paths(s1_path)

    image_bands_s2 = read_bands(s2_band_paths)
    image_bands_s1 = read_bands(s1_band_paths)
    label_band = read_band(label_path)

    for band in image_bands_s2:
        print(band.dtype()) 
    for band in image_bands_s1:
        print(band.dtype())
    print(label_band.dtype())





# -----------------------------------------------------------------------------
# ### Define main 
# -----------------------------------------------------------------------------
@log_runtime('Main')
def main(session_name, subsample):
    spark = SparkSession.builder\
        .appName(session_name)\
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()
    print(f'Spark session created: {session_name}')

    meta_schema = StructType([
        StructField('split', StringType(), True),
        StructField('s1_path', StringType(), True),
        StructField('s2_path', StringType(), True),
        StructField('label_path', StringType(), True)
    ])

    # Read metadata 
    meta = spark.read.schema(meta_schema).parquet('s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet')
    
    # Subsample dataset for gradually increasing the size of the dataset
    fractions = {"train": subsample, "test": subsample, "validation": subsample}

    meta = meta.sampleBy('split', fractions, seed=42)
   
    # Add column that holds as array all paths to the respective images for each patch 
    meta = prepare_cu_metadata(meta)

    meta = meta.limit(1) 

    meta.rdd.foreach(stack_image_array)

    spark.stop()

# -----------------------------------------------------------------------------
# ### Run main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spark session name')
    parser.add_argument('--session_name', type=str, required=True, help='Name of the Spark session')
    parser.add_argument('--subsample', type=float, required=True, help='Limit the number of images per split to process')
    args = parser.parse_args()

    main(args.session_name, args.subsample)
# -----------------------------------------------------------------------------
# ### End of script