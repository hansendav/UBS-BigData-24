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
print(f"Number of patches in the dataset: {df.count()}")

print(f"Display of the top 5 rows of the dataframe:")
df_parquet.show(5)