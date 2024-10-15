from pyspark.sql import SparkSession  

spark = SparkSession.builder\
        .master("yarn")\
        .appName("big_earth_net_to_dataframe")\
        .getOrCreate() 

df_s1 = spark.read.format("binaryFile")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/BigEarthNet-S1/")

df_s2 = spark.read.format("binaryFile")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/BigEarthNet-S2/")

df_ref_maps = spark.read.format("binaryFile")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/Reference_Maps/")


dfs = [df_s1, df_s2, df_ref_maps]
shapes = []

for df in dfs:
    print("Schema of the dataframe:")
    df.printSchema()
    print("Adding the shape of the dataframe to the list")
    rows, columns = df.count(), len(df.columns)
    shapes.append((rows, columns))

print("Shapes of the dataframes:")
print(shapes)