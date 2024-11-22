# Custom transformer to use within the ML pipeline for BigEarth ML processing 

from pyspark.sql.types import * # import all datatypes
import pyarrow.fs as fs
from pyspark.sql.functions import col, udf
from pyspark.sql import functions as f
import rasterio 
import re 
import numpy as np

from pyspark.ml import Transformer
from pypsark.ml.param.shared import HasInputCols, HasOutputCols




class extractPixels(Transformer, HasInputCols):
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
        patch_id = rows[3]
        split = rows[4]

        return s1_path, s2_path, label_path, patch_id, split


    def create_pixel_arrays(patch_path_array):

        s1_path, s2_path, label_path, patch_id, split = get_paths_from_meta(patch_path_array)

        s2_band_paths = get_band_paths(s2_path, is_s2=True)
        s1_band_paths = get_band_paths(s1_path)
        label_band_paths = get_band_paths(label_path)
        
        image_bands_s2 = read_bands(s2_band_paths)
        image_bands_s1 = read_bands(s1_band_paths)
        image_label = read_bands(label_band_paths)[0]

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
        create_pixel_arrays = udf(self.create_pixel_arrays, self.schema)

        # Apply transformation to input df  
        df = df.withColumn('pixel_arrays', create_pixel_arrays(self.input_col))

        return df 

class explode_pixel_arrays_into_df(Transformer, HasInputCols):
    def __init__(self, input_col=None, output_cols=None):
        super(extractPixels, self).__init__()
        self.input_col = input_col 
        self.output_cols = output_cols



class explode_pixel_arrays_into_df(Transformer):
    def __init__(self, input_col=None, output_cols=None):
        super(extractPixels, self).__init__():
    
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
            explode_df = udf(self.explode_to_pixel_df)

            return explode_df(df)

class create_indices(Tansformer):
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