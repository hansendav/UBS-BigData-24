# Import libraries 
from pyspark.sql import SparkSession 
from pyspark.sql.types import * # import all datatypes
import pyarrow.fs as fs
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as f
import rasterio 
import re 
import numpy as np
import argparse

# MLIB 
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols

# -----------------------------------------------------------------------------
# ### Define functions
# -----------------------------------------------------------------------------
def prepare_cu_metadata(metadata):
    metadata = metadata \
    .withColumn('s1_path', f.split(f.col('s1_path'), 's3://').getItem(1))\
    .withColumn('s2_path', f.split(f.col('s2_path'), 's3://').getItem(1))\
    .withColumn('label_path', f.split(f.col('label_path'), 's3://').getItem(1))\
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

# -----------------------------------------------------------------------------
# ### Definition custom transformers 
# ----------------------------------------------------------------------------- 
class extractPixels(Transformer):
    """
    Custom transformer to extract pixel arrays from image bands stored in the 
    metadata dataframe. 
    Input: col paths_array: array of paths to S1, S2 and label images
    Returns: col pixel_arrays: array of pixel arrays for each patch
    """
    def __init__(self):
        super(extractPixels, self).__init__()
        self.schema_pixelarray = StructType([
            StructField('VH', ArrayType(DoubleType()), True),
            StructField('VV', ArrayType(DoubleType()), True),
            StructField('B', ArrayType(LongType()), True),
            StructField('G', ArrayType(LongType()), True),
            StructField('R', ArrayType(LongType()), True),
            StructField('NIR', ArrayType(LongType()), True),
            StructField('label', ArrayType(LongType()), True)
        ])

    def get_band_paths(self, patch_path, is_s2=False):
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

    def read_band(self, band_path): 
            with rasterio.open(band_path) as src:
                band = src.read()
            return band

    def read_bands(self, band_paths):
        bands = [self.read_band(band_path) for band_path in band_paths]
        bands = [band.flatten() for band in bands]
        return bands

    def get_paths_from_meta(self, patch_path_array):
        rows = patch_path_array
        s1_path = rows[0]
        s2_path = rows[1]
        label_path = rows[2]
        patch_id = rows[3]
        split = rows[4]

        return s1_path, s2_path, label_path, patch_id, split


    def create_pixel_arrays(self, patch_path_array):

        s1_path, s2_path, label_path, patch_id, split = self.get_paths_from_meta(patch_path_array)

        s2_band_paths = self.get_band_paths(s2_path, is_s2=True)
        s1_band_paths = self.get_band_paths(s1_path)
        label_band_paths = self.get_band_paths(label_path)
        
        image_bands_s2 = self.read_bands(s2_band_paths)
        image_bands_s1 = self.read_bands(s1_band_paths)
        image_label = self.read_bands(label_band_paths)[0]

        #patch_id_array = np.repeat(patch_id, len(image_label.flatten()))
        #split_array = np.repeat(split, len(image_label.flatten()))

        row = Row('VH',
                'VV',
                'B',
                'G',
                'R',
                'NIR',
                'label')(image_bands_s1[0].tolist(),
                            image_bands_s1[1].tolist(),
                            image_bands_s2[0].tolist(),
                            image_bands_s2[1].tolist(),
                            image_bands_s2[2].tolist(),
                            image_bands_s2[3].tolist(),
                            image_label.flatten().tolist())
        return row

    def _transform(self, df):
        # set create_pixel_arrays as a UDF 
        create_pixel_arrays = udf(self.create_pixel_arrays, self.schema_pixelarray)

        # Apply transformation to input df  
        df = df.withColumn('pixel_arrays', create_pixel_arrays('paths_array'))

        return df 




class explode_pixel_arrays_into_df(Transformer):
    def __init__(self):
        super(explode_pixel_arrays_into_df, self).__init__()
    
    def explode_to_pixel_df(df): 
        to_select = ['pixel_arrays.VV',
                    'pixel_arrays.VH',
                    'pixel_arrays.B',
                    'pixel_arrays.G',
                    'pixel_arrays.R',
                    'pixel_arrays.NIR',
                    'pixel_arrays.label'
        ]
        
        df_pixels_arrays = df.select(to_select)

        df_pixels_arrays.printSchema() 
        df_pixels_arrays = df_pixels_arrays.withColumn("zipped", f.arrays_zip(
            col('VV'),
            col('VH'),
            col('B'),
            col('G'),
            col('R'),
            col('NIR'),
            col('label')
        ))

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

        def _transform(self, df):
            explode_df = self.explode_to_pixel_df(df)
            return explode_df

class create_indices(Transformer):
    def __init__(self):
        super(create_indices, self).__init__()
    
    def _transform(self, df):
        df = df\
            .withColumn('NDVI', (col('NIR') - col('R')) / (col('NIR') + col('R')))\
            .withColumn('EVI', 2.5 * (col('NIR') - col('R')) / (col('NIR') + 6 * col('R') - 7.5 * col('B') + 1))\
            .withColumn('NDWI', (col('G') - col('NIR')) / col('G') + col('NIR'))\
            .withColumn('BVI', col('G') / col('NIR'))\
            .withColumn('RRadio', col('VV') / col('VH'))\
            .withColumn('RDiff', col('VV') - col('VH'))\
            .withColumn('RSum', col('VV') + col('VH'))\
            .withColumn('RVI', 4 *col('VH') / (col('VH') + col('VV')))

        return df 

class change_label_names(Transformer):
    def __init__(self):
        super(change_label_names, self).__init__()    
        def set_spark(self, spark):
            self.dict = spark.read.csv('s3://ubs-cde/home/e2405193/bigdata/label_encoding.csv', header=True)
    def _transform(self, df):
        df = df.join(self.dict, df.label == self.ID, 'inner')\
        .drop('label')\
        .drop('DESC')\
        .drop('ID')\
        .withColumnRenamed('ID_NEW', 'label')\
        .withColumn('label', f.col('label').cast('long'))

        return df 

class custom_vector_assembler(Transformer):
    def __init__(self):
        super(custom_vector_assembler, self).__init__()

    def _transform(self, df):
        feature_cols = [col for col in df.columns if col != 'label']
        feature_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        return feature_assembler.transform(df)
# -----------------------------------------------------------------------------
# ### Define main 
# -----------------------------------------------------------------------------
def main(session_name, meta_limit):
    spark = SparkSession.builder\
        .appName(session_name)\
        .getOrCreate()
    print(f'Spark session created: {session_name}')

    # Read metadata 
    meta = spark.read.parquet('s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet')
    
    # Add column that holds as array all paths to the respective images for each patch 
    meta = prepare_cu_metadata(meta)

    # Split into train, test, validation 
    train_meta = meta.filter(meta.split == 'train') 
    val_meta = meta.filter(meta.split == 'val')
    test_meta = meta.filter(meta.split == 'test')

    # Subsample metadata in each split
    train_meta = train_meta.sample(withReplacement=False, fraction=meta_limit, seed=42)
    val_meta = val_meta.sample(withReplacement=False, fraction=meta_limit, seed=42)
    test_meta = test_meta.sample(withReplacement=False, fraction=meta_limit, seed=42)

    ## MODEL TRAINING AND EVALUATION
    

    # Feature engineering and transformation
    
    # Test intermediate outputs
    # Step 1: Apply pixel_extractor
    pixel_extractor = extractPixels()
    pixel_output = pixel_extractor.transform(train_meta)
    print("After pixel_extractor:")
    pixel_output.show(1)

    # Step 2: Apply df_transformer
    df_transformer = explode_pixel_arrays_into_df()
    df_transformed_output = df_transformer.transform(pixel_output)
    print("After df_transformer:")
    df_transformed_output.show(1)

    # Step 3: Apply indices_transformer
    indices_transformer = create_indices()
    indices_output = indices_transformer.transform(df_transformed_output)
    print("After indices_transformer:")
    indices_output.show(1)

    label_transformer.set_spark(spark)
    label_output = label_transformer.transform(indices_output)
    label_transformer = change_label_names()
    label_output = label_transformer.transform(indices_output)
    print("After label_transformer:")
    label_output.show(1)
    
    # Step 5: Apply feature_assembler
    feature_assembler = custom_vector_assembler()
    features_output = feature_assembler.transform(label_output)
    print("After feature_assembler:")
    features_output.show(1)

    


    

  

   

    """
    # Random Forest Classifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    # Pipeline setup    
    pipeline = Pipeline(stages=[pixel_extractor,
    df_transformer,
    indices_transformer,
    label_transformer,
    feature_assembler,
    rf])


    rf_model = pipeline.fit(train)
    
    preds_train = rf_model.transform(train)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(preds)

    print(f"Training set accuracy: {accuracy}")
    """


    spark.stop()

# -----------------------------------------------------------------------------
# ### Run main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spark session name')
    parser.add_argument('--session_name', type=str, required=True, help='Name of the Spark session')
    parser.add_argument('--meta_limit', type=float, required=True, help='Limit the number of images per split to process')
    args = parser.parse_args()

    main(args.session_name, args.meta_limit)
# -----------------------------------------------------------------------------
# ### End of script