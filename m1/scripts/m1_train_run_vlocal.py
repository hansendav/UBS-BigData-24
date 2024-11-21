# Import libraries 
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as f
import pyarrow.parquet as pq
import pyarrow.fs as fs
import pyarrow.csv as csv
import rasterio 
import re 
import numpy as np
import pandas as pd 
import pyspark.pandas as ps 

# MLIB 
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler


# Helper functions 
def prepare_cu_metadata(metadata):
    metadata = metadata \
    .withColumn('s1_path', f.split(f.col('s1_path'), 's3://').getItem(1)) \
    .withColumn('s2_path', f.split(f.col('s2_path'), 's3://').getItem(1)) \
    .withColumn('label_path', f.split(f.col('label_path'), 's3://').getItem(1)) \
    .withColumn(
        'patch_path_array',
        f.array(
            f.col('s1_path'),
            f.col('s2_path'),
            f.col('label_path'),
            f.col('patch_id'),
            f.col('split')
        )
    )

    return metadata

def get_band_paths(filesystem, patch_path, is_s2=False):
    if is_s2:
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
    bands = [band.flatten() for band in bands]
    return bands

def get_paths_from_meta(patch_path_array):
    rows = patch_path_array
    s1_path = rows[0]
    s2_path = rows[1]
    label_path = rows[2]
    patch_id = rows[3]
    split = rows[4]

    return s1_path, s2_path, label_path, patch_id, split

def create_pixel_arrays(patch_path_array):
    s3 = fs.S3FileSystem()

    s1_path, s2_path, label_path, patch_id, split = get_paths_from_meta(patch_path_array)

    s2_band_paths = get_band_paths(s3, s2_path, is_s2=True)
    s1_band_paths = get_band_paths(s3, s1_path)
    label_band_paths = get_band_paths(s3, label_path)
    
    image_bands_s2 = read_bands(s2_band_paths)
    image_bands_s1 = read_bands(s1_band_paths)
    image_label = read_bands(label_band_paths)[0]

    patch_id_array = np.repeat(patch_id, len(image_label.flatten()))
    split_array = np.repeat(split, len(image_label.flatten()))

    row = Row('VH_px',
              'VV_px',
              'B_px',
              'G_px',
              'R_px',
              'NIR_px',
              'label_px',
              'patch_id_px',
              'split_px')(image_bands_s1[0].tolist(),
                          image_bands_s1[1].tolist(),
                          image_bands_s2[0].tolist(),
                          image_bands_s2[1].tolist(),
                          image_bands_s2[2].tolist(),
                          image_bands_s2[3].tolist(),
                          image_label.flatten().tolist(),
                          patch_id_array.tolist(),
                          split_array.tolist())
    return row

# make UDF 
schema = StructType([
    StructField('VH', ArrayType(DoubleType()), True),
    StructField('VV', ArrayType(DoubleType()), True),
    StructField('B', ArrayType(LongType()), True),
    StructField('G', ArrayType(LongType()), True),
    StructField('R', ArrayType(LongType()), True),
    StructField('NIR', ArrayType(LongType()), True),
    StructField('label', ArrayType(LongType()), True),
    StructField('patch_id', ArrayType(StringType()), True),
    StructField('split', ArrayType(StringType()), True)
])

create_pixel_arrays_udf = udf(create_pixel_arrays, schema)

def main():
    spark = SparkSession.builder\
        .appName("Test")\
        .master("local[*]")\
        .getOrCreate()

    print('Spark session created')

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    s3 = fs.S3FileSystem()

    meta = pq.read_table('s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet').to_pandas()
    meta = meta.iloc[:5]
    meta = spark.createDataFrame(meta)

    meta = prepare_cu_metadata(meta)
    
    meta = meta.withColumn('pixeldata', create_pixel_arrays_udf('patch_path_array'))

    meta.show(1)
    meta.printSchema()
    
    to_select = ['pixeldata.VV',
                'pixeldata.VH',
                'pixeldata.B',
                'pixeldata.G',
                'pixeldata.R',
                'pixeldata.NIR',
                'pixeldata.label',
                'pixeldata.patch_id',
                'pixeldata.split']
    test_df = meta.select(to_select)
    explode_df = test_df.select(f.posexplode(col('VV')).alias('pos', 'VV'))\
        .join(test_df.select(f.posexplode(col('VH')).alias('pos', 'VH')), 'pos')\
        .join(test_df.select(f.posexplode(col('B')).alias('pos', 'B')), 'pos')\
        .join(test_df.select(f.posexplode(col('G')).alias('pos', 'G')), 'pos')\
        .join(test_df.select(f.posexplode(col('R')).alias('pos', 'R')), 'pos')\
        .join(test_df.select(f.posexplode(col('NIR')).alias('pos', 'NIR')), 'pos')\
        .join(test_df.select(f.posexplode(col('label')).alias('pos', 'label')), 'pos')\
        .join(test_df.select(f.posexplode(col('patch_id')).alias('pos', 'patch_id')), 'pos')\
        .join(test_df.select(f.posexplode(col('split')).alias('pos', 'split')), 'pos')\
        .drop('pos')
    
    print(explode_df.count())    
    explode_df.show(1)
    print('Done')
    spark.stop()

if __name__ == "__main__":
    main()