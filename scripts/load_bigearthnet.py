from pyspark import SparkContext 
sc = SparkContext(master='local[4]') # specify here based on the cluster 
sc.appName = 'load_bigearthnet_to_dataframe'
print(sc.appName)