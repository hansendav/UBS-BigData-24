from pyspark.sql import SparkSession  

spark = SparkSession.builder\
        .master("yarn")\
        .appName("big_earth_net_to_dataframe")\
        .getOrCreate() 

df = spark.read.format("binaryFile")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/")


print("Schema of the dataframe:")
df.printSchema()

print("Show the dataframe:")
df.show() 

print("Dataframe shape") 
print(f"Number of rows: {df.count()}")
print(f"Number of columns: {len(df.columns)}")