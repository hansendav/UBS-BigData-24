from pyspark.sql import SparkSession

spark = SparkSession.builder\
        .master("yarn")\
        .appName("read_parquet_file_bigearth")\
        .getOrCreate()

df_s1 = spark.read\
    .format("binaryFile")\
    .option("recursiveFileLookup", "true")\
    .option("pathGlobFilter", "*.tif")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S1/")\
    .select("path")

df_s2 = spark.read.format("binaryFile")\
    .option("recursiveFileLookup", "true")\
    .option("pathGlobFilter", "*.tif")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/")\
    .select("path")


df_ref_maps = spark.read.format("binaryFile")\
    .option("recursiveFileLookup", "true")\
    .option("pathGlobFilter", "*.tif")\
    .load("s3://ubs-datasets/bigearthnet/Reference_Maps/")\
    .select("path")

df_parquet = spark.read.parquet("s3://ubs-datasets/bigearthnet/metadata.parquet")\
            .limit(5)

df_parquet.printSchema()
print(f"Number of patches in the dataset: {df_parquet.count()}")

print(f"Display of the top 5 rows of the dataframe:")
df_parquet.show(5)

"""
root
 |-- patch_id: string (nullable = true)
 |-- labels: array (nullable = true)
 |    |-- element: string (containsNull = true)
 |-- split: string (nullable = true)
 |-- country: string (nullable = true)
 |-- s1_name: string (nullable = true)
 |-- s2v1_name: string (nullable = true)
 |-- contains_seasonal_snow: boolean (nullable = true)
 |-- contains_cloud_or_shadow: boolean (nullable = true)

Number of patches in the dataset: 5
Display of the top 5 rows of the dataframe:
+--------------------+--------------------+-----+-------+--------------------+--------------------+----------------------+------------------------+
|            patch_id|              labels|split|country|             s1_name|           s2v1_name|contains_seasonal_snow|contains_cloud_or_shadow|
+--------------------+--------------------+-----+-------+--------------------+--------------------+----------------------+------------------------+
|S2A_MSIL2A_201706...|[Arable land, Bro...| test|Austria|S1B_IW_GRDH_1SDV_...|S2A_MSIL2A_201706...|                 false|                   false|
|S2A_MSIL2A_201706...|[Arable land, Bro...| test|Austria|S1B_IW_GRDH_1SDV_...|S2A_MSIL2A_201706...|                 false|                   false|
|S2A_MSIL2A_201706...|[Arable land, Bro...| test|Austria|S1B_IW_GRDH_1SDV_...|S2A_MSIL2A_201706...|                 false|                   false|
|S2A_MSIL2A_201706...|[Broad-leaved for...| test|Austria|S1B_IW_GRDH_1SDV_...|S2A_MSIL2A_201706...|                 false|                   false|
|S2A_MSIL2A_201706...|[Broad-leaved for...| test|Austria|S1B_IW_GRDH_1SDV_...|S2A_MSIL2A_201706...|                 false|                   false|
+--------------------+--------------------+-----+-------+--------------------+--------------------+----------------------+------------------------+
"""