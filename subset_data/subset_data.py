#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getpass
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, avg, count, countDistinct


def main(spark, netID, fraction):
    print('An innocent quokka is fetching data from hdfs')

    # Load the boats.txt and sailors.json data into DataFrame
    interaction_parquet = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    interaction_parquet.createOrReplaceTempView('interaction_parquet')
    
    print('An innocent quokka is printing schema...')
    interaction_parquet.printSchema()
#     root
#  |-- user_id: string (nullable = true)
#  |-- count: long (nullable = true)
#  |-- track_id: string (nullable = true)
#  |-- __index_level_0__: long (nullable = true)
    
    interaction001 = interaction_parquet.sample(withReplacement=False, fraction = fraction, seed = 857)
    interaction001.write.mode('overwrite').parquet(f'cf_train_{str(fraction)}.parquet')


    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final_project').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    
    fraction = float(sys.argv[-1])
    # Call our main routine
    main(spark, netID, fraction)
    