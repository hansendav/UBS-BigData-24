# BigData Project M1
# Step 1: Create train, test and validation dataframes from BigEarthNet
# Reads custom metadata.parquet file from S3.
# Processes train, test and validation images and labels as follows: 
# For train, test, validation respectively:
# Get image paths from metadata.parquet; 
# Get label paths from metadata.parquet;
# Read S1 bands and perform band math; 
# Read S2 bands and perform band math;
# Read and store labels 
# Store dataframe as parquet file in S3 


# Import libraries 
from pyspark.sql import SparkSession 
from pyspark.sql.types import StructType, StructField, LongType, DoubleType, StringType
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as f
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



def get_s2_band_paths(filesystem, patch_path): 
    files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
    file_paths = ['s3://' + file.path for file in files_info if file.is_file and re.search(r'_B(0[2348]).tif$', file.path)]
    return file_paths

def get_s1_band_paths(filesystem, patch_path):
    # only VV and VH bands available -> no filtering needed
    files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
    file_paths = ['s3://' + file.path for file in files_info if file.is_file] 
    return file_paths

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

def get_paths_from_meta(meta_path_row):
    s1_path = meta_path_row.collect()[0].path
    s2_path = meta_path_row.collect()[1].path
    label_path = meta_path_row.collect()[2].path
    patch_id = meta_path_row.collect()[3].path
    split = meta_path_row.collect()[4].path

    return s1_path, s2_path, label_path, patch_id, split


def create_image_dataframe(spark, s3, meta_path_row):
    s1_path, s2_path, label_path, patch_id, split= get_paths_from_meta(meta_path_row)

    s2_band_paths = get_s2_band_paths(s3, s2_path)
    s1_band_paths = get_s1_band_paths(s3, s1_path)
    label_band_paths = get_label_path(s3, label_path)
    
    image_bands_s2 = read_bands(s2_band_paths)
    image_bands_s1 = read_bands(s1_band_paths)
    image_label = read_bands(label_band_paths)[0]

    schema = StructType([
    StructField('VH', DoubleType(), True),
    StructField('VV', DoubleType(), True),
    StructField('B', LongType(), True),
    StructField('G', LongType(), True),
    StructField('R', LongType(), True),
    StructField('NIR', LongType(), True),
    StructField('label', LongType(), True),
    StructField('patch_id', StringType(), True),
    StructField('split', StringType(), True)
    ])


    # Need to create pdf first to avoid dtype error 
    # not solved yet -> possible bottleneck here
    pdf = pd.DataFrame(
    {
        'VH': image_bands_s1[0],
        'VV': image_bands_s1[1],
        'B': image_bands_s2[0],
        'G': image_bands_s2[1],
        'R': image_bands_s2[2],
        'NIR': image_bands_s2[3],
        'label': image_label.flatten(),
        'patch_id': np.repeat(patch_id, len(image_label.flatten())),
        'split': np.repeat(split, len(image_label.flatten()))
    })
    
    df_spark = spark.createDataFrame(pdf, schema=schema)

    df_spark = df_spark\
        .withColumn('NDVI', (col('NIR') - col('R')) / (col('NIR') + col('R')))\
        .withColumn('EVI', 2.5 * (col('NIR') - col('R')) / (col('NIR') + 6 * col('R') - 7.5 * col('B') + 1))\
        .withColumn('NDWI', (col('G') - col('NIR')) / col('G') + col('NIR'))\
        .withColumn('BVI', col('G') / col('NIR'))\
        .withColumn('RRadio', col('VV') / col('VH'))\
        .withColumn('RDiff', col('VV') - col('VH'))\
        .withColumn('RSum', col('VV') + col('VH'))\
        .withColumn('RVI', 4 *col('VH') / (col('VH') + col('VV')))

    return df_spark

create_image_dataframe_udf = udf(create_image_dataframe)


def save_dataframe(dataframe, output_dir):
    file_path = f"{output_dir}/{path1}_{path2}_{path3}.parquet"
    df.write.parquet(file_path)
    return file_path


def main():
    spark = SparkSession.builder\
        .appName("Test")\
        .master("local[*]")\
        .config("spark.eventLog.dir", "./log/") \
        .getOrCreate()
    
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # Read metadata.parquet file from S3
    meta = spark.read.parquet('ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet')
    meta.show(1)

    label_dict = spark.read.csv('ubs-cde/home/e2405193/bigdata/label_encoding.csv', header=True)

    
    """
    s3 = fs.S3FileSystem()

    meta = pq.read_table('s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet').to_pandas()
    meta = meta.iloc[:1]
    meta = spark.createDataFrame(meta)

    # Specify the path to your CSV file in S3
    s3_path_dict = 'ubs-cde/home/e2405193/bigdata/label_encoding.csv'

    # Open the CSV file from S3 and read it into a table
    with s3.open_input_stream(s3_path_dict) as file:
        label_dict = csv.read_csv(file)

    label_dict = label_dict.to_pandas()
    label_dict = spark.createDataFrame(label_dict)


    meta = meta \
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

    paths = meta.select(f.explode(meta.patch_path_array).alias('path'))

    s1_path, s2_path, label_path, patch_id, split= get_paths_from_meta(paths)

    s2_band_paths = get_s2_band_paths(s3, s2_path)
    s1_band_paths = get_s1_band_paths(s3, s1_path)
    label_band_paths = get_label_path(s3, label_path)
    
    image_bands_s2 = read_bands(s2_band_paths)
    image_bands_s1 = read_bands(s1_band_paths)
    image_label = read_bands(label_band_paths)[0]

    print(image_label)


    df_image = create_image_dataframe(spark, s3, paths)

    df_image = df_image.join(label_dict, df_image.label == label_dict.ID, 'inner')\
        .drop('label')\
        .drop('DESC')\
        .withColumnRenamed('ID_NEW', 'label')


    feature_cols = [col for col in df_image.columns if col not in ['split', 'label', 'patch_id']]
    feature_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    #scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_numerical_features") # output


    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    pipeline = Pipeline(stages=[feature_assembler, rf])


    rf_model = pipeline.fit(df_image)

    preds = rf_model.transform(df_image)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(preds)

    print(f"Training set accuracy: {accuracy}")
    
    #df_image_pd = df_image.toPandas()
    #df_image_pd.info(memory_usage='deep')
    #df_image.show()
    #df_image.printSchema()  
    """
    spark.stop()
if __name__ == "__main__":
    main()


