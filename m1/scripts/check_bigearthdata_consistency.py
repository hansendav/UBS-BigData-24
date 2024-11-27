from pyspark.sql import SparkSession 
from pyspark.sql import functions as f 
from pyspark.sql.types import * 
from pyspark.sql.functions import pandas_udf, PandasUDFType


import pyarrow.parquet as pq
import pyarrow.fs as fs
import re 

import numpy as np


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

def main(): 
    spark = SparkSession.builder\
        .appName('check_bigearthdata_consistency')\
        .config('pandas.arrow.enabled', 'true')\
        .config('spark.executor.instances', '4')\
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

    meta = sample.withColumn('ns2bands', get_number_of_bands_udf(f.col('s2_path')))\
        .withColumn('ns1bands', get_number_of_bands_udf(f.col('s1_path')))\
        .withColumn('nlabelbands', get_number_of_bands_udf(f.col('label_path')))

    missing_bands = meta.filter(f.col('ns2bands') != 12 or f.col('ns1bands') != 2 or f.col('nlabelbands') != 1)

    missing_bands.write.mode('overwrite').parquet('s3://ubs-cde/home/e2405193/bigdata/missing_bands.parquet')

    spark.stop()
    

if __name__ == '__main__':
    main()

