from pyspark.sql import SparkSession 
from pyspark.sql import functions as f 
from pyspark.sql.types import * 
from pyspark.sql.functions import pandas_udf, PandasUDFType


import pyarrow.parquet as pq
import pyarrow.fs as fs
import re 

import numpy as np


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


def prepare_cu_metadata(metadata):
    metadata = metadata \
    .withColumn('s1_path', f.split(f.col('s1_path'), 's3://').getItem(1))\
    .withColumn('s2_path', f.split(f.col('s2_path'), 's3://').getItem(1))\
    .withColumn('label_path', f.split(f.col('label_path'), 's3://').getItem(1))

    return metadata

def get_number_of_bands(patch_path, is_s2=False):
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


        nfiles = len(file_paths)
        print(f'Number of files: {nfiles}')
        
        return nfiles

get_number_of_bands_udf = f.udf(get_number_of_bands, IntegerType())

@log_runtime('Main - Check BigEarthData Consistency')
def main(): 
    spark = SparkSession.builder\
        .appName('check_bigedata_con')\
        .master('yarn')\
        .getOrCreate()

    # Meta schema definiton
    meta_schema = StructType([
        StructField('patch_id', StringType(), True),
        StructField('s1_path', StringType(), True),
        StructField('s2_path', StringType(), True),
        StructField('label_path', StringType(), True)])

    # Read metadata 
    meta = spark.read.schema(meta_schema).parquet('s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet')
    
    meta = prepare_cu_metadata(meta)

    meta = meta.limit(10000)

    meta = meta.withColumn('ns2bands', get_number_of_bands_udf(f.col('s2_path')))\
        .withColumn('ns1bands', get_number_of_bands_udf(f.col('s1_path')))\
        .withColumn('nlabelbands', get_number_of_bands_udf(f.col('label_path')))
    


    missing_bands = meta.filter((f.col('ns2bands') != 12) | (f.col('ns1bands') != 2) | (f.col('nlabelbands') != 1))
    missing_bands.write.parquet('s3://ubs-cde/home/e2405193/bigdata/missing_bands_patch_id.parquet', mode='overwrite')

    spark.stop()
    
if __name__ == '__main__':
    main()