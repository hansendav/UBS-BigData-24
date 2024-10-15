from pyspark.sql import SparkSession  

spark = SparkSession.builder\
        .master("yarn")\
        .appName("big_earth_net_to_dataframe")\
        .getOrCreate() 

def load_data_frame(path, format="binaryFile", type='.tif', recursive=True):
    """Loads a dataframe from a given path with a given format and type.
    """
    if recursive == True:
        df = spark.read.format(format)\
            .option("recursiveFileLookup")\
            .option("pathGlobFilter", f"*{type}")\
            .load(path)
    else: 
        df = spark.read.format(format)\
            .option("pathGlobFilter", f"*{type}")\
            .load(path)
    return df

df_s1 = spark.read.format("binaryFile")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S1/")

df_s2 = spark.read.format("binaryFile")\
    .load("s3://ubs-datasets/bigearthnet/BigEarthNet-S2/")

df_ref_maps = spark.read.format("binaryFile")\
    .load("s3://ubs-datasets/bigearthnet/Reference_Maps/")


dfs = [df_s1, df_s2, df_ref_maps]
shapes = []

for df in dfs:
    print("Schema of the dataframe:")
    df.printSchema()
    print("Adding the shape of the dataframe to the list")
    rows, columns = df.count(), len(df.columns) # cols does not make any sense here -> is always the same 
    shapes.append((rows, columns))

print("Shapes of the dataframes:")
print(shapes)