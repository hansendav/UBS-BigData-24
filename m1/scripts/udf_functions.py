
# gets the patch_path_array column of the meta dataframe 
# explodes into a new row for each element in the array
# to be used to get image paths for further processing 
meta.select(f.explode(meta.patch_path_array).alias('path'))


# Position explode into new dataframe such that array elements fit into single rows 
exploded_df = (
    df.select(posexplode(col("a1")).alias("pos", "a1"))
    .join(df.select(posexplode(col("a2")).alias("pos", "a2")), on="pos")
    .join(df.select(posexplode(col("a3")).alias("pos", "a3")), on="pos")
    .drop("pos")
)

# => this is the way to do it! 

def get_paths_from_meta(meta_path_row):
    s1_path = meta_path_row.collect()[0].path
    s2_path = meta_path_row.collect()[1].path
    label_path = meta_path_row.collect()[2].path
    patch_id = meta_path_row.collect()[3].path
    split = meta_path_row.collect()[4].path

    return s1_path, s2_path, label_path, patch_id, split


def get_band_paths(filesystem, patch_path, is_s2=False): 
    """
    Extracts image band paths from a given patch path. 
    For Sentinel-2, only bands B02, B03, B04, B08 are extracted.
    """
    if is_s2 == True:
        files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
        file_paths = ['s3://' + file.path for file in files_info if file.is_file and re.search(r'_B(0[2348]).tif$', file.path)]
    else:
        files_info = filesystem.get_file_info(fs.FileSelector(patch_path, recursive=True))
        file_paths = ['s3://' + file.path for file in files_info if file.is_file]
    return file_paths

def read_band(band_path): 
    """
    Reads a single band from a given path.
    Using rasterio open to np.array 
    """
    with rasterio.open(band_path) as src:
        band = src.read()
    return band

def read_bands(band_paths):
    """
    Given image band paths as input, reads all bands and flattens. 
    Resulting in list of np.arrays for containing all pixels.
    """
    bands = [read_band(band_path) for band_path in band_paths]
    bands = [band.flatten() for band in bands]
    return bands




def aggregate_pixel_values(spark, s3, meta_path_row):
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
