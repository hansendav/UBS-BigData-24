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

    return s1_path, s2_path, label_path


def create_pixel_arrays(patch_path_array):
    s3 = fs.S3FileSystem()

    s1_path, s2_path, label_path = get_paths_from_meta(patch_path_array)

    s2_band_paths = get_band_paths(s2_path, is_s2=True)
    s1_band_paths = get_band_paths(s1_path)
    label_band_paths = get_band_paths(label_path)
    
    image_bands_s2 = read_bands(s2_band_paths)
    image_bands_s1 = read_bands(s1_band_paths)
    image_label = read_bands(label_band_paths)[0]

    row = Row('VH',
              'VV',
              'B',
              'G',
              'R',
              'NIR',
              'label_p')(image_bands_s1[0].tolist(),
                          image_bands_s1[1].tolist(),
                          image_bands_s2[0].tolist(),
                          image_bands_s2[1].tolist(),
                          image_bands_s2[2].tolist(),
                          image_bands_s2[3].tolist(),
                          image_label.flatten().tolist())
    return row

# make UDF 
schema = StructType([
    StructField('VH', ArrayType(DoubleType()), True),
    StructField('VV', ArrayType(DoubleType()), True),
    StructField('B', ArrayType(LongType()), True),
    StructField('G', ArrayType(LongType()), True),
    StructField('R', ArrayType(LongType()), True),
    StructField('NIR', ArrayType(LongType()), True),
    StructField('label', ArrayType(LongType()), True)
])

create_pixel_arrays_udf = udf(create_pixel_arrays, schema)

def explode_to_pixel_df(meta_df): 
    to_select = ['pixel_arrays.VV',
                'pixel_arrays.VH',
                'pixel_arrays.B',
                'pixel_arrays.G',
                'pixel_arrays.R',
                'pixel_arrays.NIR',
                'pixel_arrays.label']
            

    df_pixels_arrays = meta_df.select(to_select)

    df_pixels_arrays.printSchema() 
    df_pixels_arrays = df_pixels_arrays.withColumn("zipped", f.arrays_zip(
        col('VV'),
        col('VH'),
        col('B'),
        col('G'),
        col('R'),
        col('NIR'),
        col('label')))

    # Explode zipped arrays to ensure that each pixels is a row 
    # exactly once
    explode_df = df_pixels_arrays.select(f.explode(col('zipped')).alias('zipped'))\
        .select(
            col('zipped.VV').alias('VV'),
            col('zipped.VH').alias('VH'),
            col('zipped.B').alias('B'),
            col('zipped.G').alias('G'),
            col('zipped.R').alias('R'),
            col('zipped.NIR').alias('NIR'),
            col('zipped.label').alias('label')
        )

    return explode_df


# -----------------------------------------------------------------------------
# ### Define main 
# -----------------------------------------------------------------------------
def main(session_name, meta_limit):
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
    fractions = {"train": 0.1, "test": 0.1, "val": 0.1}

    meta = meta.sampleBy('split', fractions, seed=42)
    meta = meta.repartition(100, 'split')
    # Add column that holds as array all paths to the respective images for each patch 
    meta = prepare_cu_metadata(meta)

    # Split into train, test, validation 
    train_meta = meta.filter(meta.split == 'train') 
    val_meta = meta.filter(meta.split == 'val')
    test_meta = meta.filter(meta.split == 'test')

    
    # Apply UDF to create pixel arrays for each patch
    meta = meta.withColumn('pixel_arrays', create_pixel_arrays_udf('paths_array'))
    
    # Explode arrays to create pixel dataframe for training 
    df_pixels = explode_to_pixel_df(meta)

    ## MODEL TRAINING AND EVALUATION
    
    # Import label dictionary 
    label_dict = spark.read.csv('s3://ubs-cde/home/e2405193/bigdata/label_encoding.csv', header=True)

    # Adapt label names based on custom dictionary 
    df_pixels = df_pixels.join(label_dict, df_pixels.label == label_dict.ID, 'inner')\
        .drop('label')\
        .drop('DESC')\
        .drop('ID')\
        .withColumnRenamed('ID_NEW', 'label')\
        .withColumn('label', f.col('label').cast('long'))

    df_pixels.show(2)
    df_pixels.printSchema()

    
    # 
    # Train, Validation, Test splits    
    train = df_pixels.filter(df_pixels.split == 'train')
    val = df_pixels.filter(df_pixels.split == 'val')
    test = df_pixels.filter(df_pixels.split == 'test')

    # Add feature engineering here before assembling all into features column

    # Feature selection and assembling 
    feature_cols = [col for col in df_pixels.columns if col not in ['split', 'label', 'patch_id']]
    feature_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Random Forest Classifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    # Pipeline setup    
    pipeline = Pipeline(stages=[feature_assembler, rf])

    rf_model = pipeline.fit(train)

    preds_train = rf_model.transform(train)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(preds)

    print(f"Training set accuracy: {accuracy}")

    spark.stop()

# -----------------------------------------------------------------------------
# ### Run main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spark session name')
    parser.add_argument('--session_name', type=str, required=True, help='Name of the Spark session')
    parser.add_argument('--meta_limit', type=float, required=True, help='Limit the number of images to process')
    args = parser.parse_args()

    main(args.session_name, args.meta_limit)
# -----------------------------------------------------------------------------
# ### End of script