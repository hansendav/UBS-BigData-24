from pyspark.sql import SparkSession 
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as f
import pyarrow.parquet as pq
import pyarrow.fs as fs
import rasterio 
import re 
import numpy as np
import pandas as pd 
import pyspark.pandas as ps 
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler


# Functions 
def set_up_spark_session():
    spark = SparkSession.builder\
        .appName("Test")\
        .master("local[*]")\
        .getOrCreate()

    return spark


# Meta schema
schema = StructType([
    StructField('patch_id', StringType(), True),
    StructField('label', ArrayType(StringType()), True),
    StructField('split', StringType(), True),
    StructField('country', StringType(), True),
    StructField('s1_name', StringType(), True),
    StructField('s2v1_name', StringType(), True),
    StructField('contains_seasonal_snow', BooleanType(), True),
    StructField('contains_cloud_or_shadow', BooleanType(), True),
    StructField('patch_id_path', StringType(), True),
    StructField('patch_id_path_s1', StringType(), True),
    StructField('s1_path', StringType(), True),
    StructField('s2_path', StringType(), True),
    StructField('label_path', StringType(), True)
])


def get_label_path(filesystem, patch_path):
    files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
    file_paths = ['s3://' + file.path for file in files_info if file.is_file]
    return file_paths

def read_band(band_path): 
        with rasterio.open(band_path) as src:
            band = src.read()
        return band

def read_bands(band_paths):
    bands = [read_band(band_path) for band_path in band_paths]
    bands = [band.flatten() for band in bands]
    return bands

def main(): 
    spark = set_up_spark_session()
    sc = spark.sparkContext
    sc.version

    spark.stop()