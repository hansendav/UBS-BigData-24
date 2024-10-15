from pyspark import SparkContext 

spark = SparkSession.builder
    .master("local")
    .appName("big_earth_net_to_dataframe")
    .getOrCreate() 

df = spark.read.format("binaryFile").\
    load("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57_B01.tif")


df.printSchema()
df.show()