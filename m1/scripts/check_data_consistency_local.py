import pyarrow.parquet as pq
import pyarrow.fs as fs
import re 

import numpy as np
import time 

def log_runtime(task_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_time_formatted = time.strftime("%H:%M:%S", time.localtime(start_time))
            print(f"{task_name} started at {start_time_formatted}")
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_time_formatted = time.strftime("%H:%M:%S", time.localtime(end_time))
            print(f"{task_name} finished at {end_time_formatted}")
            
            runtime = end_time - start_time
            hours, rem = divmod(runtime, 3600)
            minutes, seconds = divmod(rem, 60)
            runtime_formatted = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
            print(f"Runtime of task {task_name}: {runtime_formatted}")
            
            return result 
        return wrapper
    return  decorator

def get_number_of_bands(patch_path, is_s2=False):
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


        nfiles = len(file_paths)
        return nfiles

@log_runtime('Main')
def main():
    s3 = fs.S3FileSystem()
    path_to_metadata = 's3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet'

    # Read parquet file to pandas df
    meta = pq.read_table(path_to_metadata).to_pandas()

    # Adapt paths for S3FileSystem listing 
    meta['s1_path'] = meta['s1_path'].str.split('s3://').str[1]
    meta['s2_path'] = meta['s2_path'].str.split('s3://').str[1]
    meta['label_path'] = meta['label_path'].str.split('s3://').str[1]

    meta.drop(columns=[
        'labels',
        'split',
        'country',
        's1_name',
        's2v1_name',
        'contains_seasonal_snow',
        'contains_cloud_or_shadow',
        'patch_id_path',
        'patch_id_path_s1'
        ],
        inplace=True)



    meta['s1_band_count'] = meta['s1_path'].apply(lambda x: get_number_of_bands(x))
    print(f'S1 band count finished')
    meta['s2_band_count'] = meta['s2_path'].apply(lambda x: get_number_of_bands(x))
    print(f'S2 band count finished')
    meta['label_band_count'] = meta['label_path'].apply(lambda x: get_number_of_bands(x))
    print(f'Label band count finished')

    meta = meta[meta['s1_band_count'] != 2 | meta['s2_band_count'] != 12 | meta['label_band_count'] != 1]

    # Save to parquet 
    pq.write_table(pq.Table.from_pandas(meta), 's3://ubs-cde/home/e2405193/bigdata/meta_with_band_counts.parquet')


if __name__ == '__main__':
    main()

