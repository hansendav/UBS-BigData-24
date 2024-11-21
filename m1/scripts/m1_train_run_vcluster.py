# Import libraries 
from pyspark.sql import SparkSession 
from pyspark.sql.types import * # import all datatypes
import pyarrow.fs as fs
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as f
import rasterio 
import re 
import numpy as np
#import pyspark.pandas as ps 

# MLIB 
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import argparse

# -----------------------------------------------------------------------------
# ### Define functions
# ----------------------------------------------------------------------------- 
def prepare_cu_metadata(metadata):
    metadata = metadata \
    .withColumn('s1_path', f.split(f.col('s1_path'), 's3://').getItem(1)) \
    .withColumn('s2_path', f.split(f.col('s2_path'), 's3://').getItem(1)) \
    .withColumn('label_path', f.split(f.col('label_path'), 's3://').getItem(1)) \
    .withColumn(
        'paths_array',
        f.array(
            f.col('s1_path'),
            f.col('s2_path'),
            f.col('label_path'),
            f.col('patch_id'),
            f.col('split')
        )
    )

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

    s2_band_paths = get_band_paths(s2_path, is_s2=True)
    s1_band_paths = get_band_paths(s1_path)
    label_band_paths = get_band_paths(label_path)
    
    image_bands_s2 = read_bands(s2_band_paths)
    image_bands_s1 = read_bands(s1_band_paths)
    image_label = read_bands(label_band_paths)[0]

    patch_id_array = np.repeat(patch_id, len(image_label.flatten()))
    split_array = np.repeat(split, len(image_label.flatten()))

    row = Row('VH',
              'VV',
              'B',
              'G',
              'R',
              'NIR',
              'label_p',
              'patch_id',
              'split')(image_bands_s1[0].tolist(),
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

def explode_to_pixel_df(meta_df): 
    to_select = ['pixel_arrays.VV',
                'pixel_arrays.VH',
                'pixel_arrays.B',
                'pixel_arrays.G',
                'pixel_arrays.R',
                'pixel_arrays.NIR',
                'pixel_arrays.label',
                'pixel_arrays.patch_id',
                'pixel_arrays.split']

    df_pixels_arrays = meta_df.select(to_select)
    explode_df = df_pixels_arrays.select(f.posexplode(col('VV')).alias('pos', 'VV'))\
        .join(df_pixels_arrays.select(f.posexplode(col('VH')).alias('pos', 'VH')), 'pos')\
        .join(df_pixels_arrays.select(f.posexplode(col('B')).alias('pos', 'B')), 'pos')\
        .join(df_pixels_arrays.select(f.posexplode(col('G')).alias('pos', 'G')), 'pos')\
        .join(df_pixels_arrays.select(f.posexplode(col('R')).alias('pos', 'R')), 'pos')\
        .join(df_pixels_arrays.select(f.posexplode(col('NIR')).alias('pos', 'NIR')), 'pos')\
        .join(df_pixels_arrays.select(f.posexplode(col('label')).alias('pos', 'label')), 'pos')\
        .join(df_pixels_arrays.select(f.posexplode(col('patch_id')).alias('pos', 'patch_id')), 'pos')\
        .join(df_pixels_arrays.select(f.posexplode(col('split')).alias('pos', 'split')), 'pos')\
        .drop('pos')


    return explode_df
    


# -----------------------------------------------------------------------------
# ### Define main 
# -----------------------------------------------------------------------------
def main(session_name):
    spark = SparkSession.builder\
        .appName(session_name)\
        .getOrCreate()
    print(f'Spark session created: {session_name}')

    # Read metadata 
    meta = spark.read.parquet('s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet')
    
    meta = meta.limit(5)
    
    # Add column that holds as array all paths to the respective images for each patch 
    meta = prepare_cu_metadata(meta)
    
    # Apply UDF to create pixel arrays for each patch
    meta = meta.withColumn('pixel_arrays', create_pixel_arrays_udf('paths_array'))
    
    # Explode arrays to create pixel dataframe for training 
    df_pixels = explode_to_pixel_df(meta)

    df_pixels.printSchema()
    print(df_pixels.count())

    spark.stop()

# -----------------------------------------------------------------------------
# ### Run main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spark session name')
    parser.add_argument('--session_name', type=str, required=True, help='Name of the Spark session')
    args = parser.parse_args()

    main(args.session_name)
# -----------------------------------------------------------------------------
# ### End of script