# This script is used to test the workflow of the mlib package and PySpark 
from pyspark.sql import SparkSession 
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
import time

# Decorators

def log_runtime(task_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.localtime()
            start_time_formated = time.strftime("%H:%M:%S". start_time)
            print(f"{task_name} started at {start_time_formated}")
            result = func(*args, **kwargs)
            end_time = time.localtime()
            end_time_formated = time.strftime("%H:%M:%S". end_time)
            print(f"{task_name} finished at {end_time_formated}")
            runtime = end_time - start_time
            runtime_formated = time.strftime("%H:%M:%S", runtime)
            print(f"Runtime task {task_name}: {runtime_formated}")
            return result 
        return wrapper
    return  decorator

def print_start_finish(whatstarted_message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"{whatstarted_message} started")
            result = func(*args, **kwargs)
            print(f"{whatstarted_message} finished")
            return result 
        return wrapper 
    return decorator


@log_runtime('CO2-regression-model')
def main():
# Set up session
    spark = SparkSession.builder\
        .appName("CO2-regression-model")\
        .getOrCreate()

    # Load the data
    path_to_file = 's3://ubs-datasets/CO2/CO2_Emissions_Canada.csv'
    df = spark.read.csv(path_to_file,
                        sep=',',
                        header=True,
                        inferSchema=True)


    train, test = df.randomSplit([0.8, 0.2])


    # define the column dtypes for further processing
    categorical_cols = [field for (field, dtype) in train.dtypes if dtype == "string"]
    numerical_cols = [field for (field, dtype) in train.dtypes if dtype in ["int", "double"]]

    # index and encode the categorical columns
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid='keep') for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_ohe") for col in categorical_cols]

    # Set numerical assembler for StandardScaling 
    numerical_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical_features")
    scaler = StandardScaler(inputCol="numerical_features", outputCol="scaled_numerical_features") # output

    # assemble all features
    assembler = VectorAssembler(inputCols=[col + "_ohe" for col in categorical_cols] + ['scaled_numerical_features'], outputCol="features")

    # -> taking again all features as featuresColumns (the general assembler)
    lr = LinearRegression(featuresCol="features", labelCol="CO2")

    # indexers and encoders are lists!!! thats why they are not within the brackets 
    pipeline = Pipeline(stages= indexers + encoders + [numerical_assembler, scaler, assembler, lr])


    # fit the model 
    @print_start_finish("Model training")
    def train_model(pipeline, trainset):
        lr_model = pipeline.fit(train)
        return lr_model

    lr_model = train_model(pipeline, train)

    #  make predictions 
    preds = lr_model.transform(test)

    # evaluate the model 
    evaluator = RegressionEvaluator(labelCol='CO2', predictionCol='prediction', metricName='rmse')
    rmse = evaluator.evaluate(preds)

    # print RMSE of the model 
    print(f"The RMSE of the simple model is: {rmse}")

    spark.stop()
    print(f"Spark session stopped")

if __name__ == "__main__":
    main()