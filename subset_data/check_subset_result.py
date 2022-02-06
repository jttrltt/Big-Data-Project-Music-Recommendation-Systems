#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getpass
import sys
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, avg, count, countDistinct


def main(spark, netID, infile):
    print('An innocent quokka is fetching data from hdfs')

    # Load the boats.txt and sailors.json data into DataFrame
    interaction001 = spark.read.parquet(f'hdfs://horton.hpc.nyu.edu:8020/user/yd1008/{infile}.parquet')
    interaction001.createOrReplaceTempView('interaction001')
    
    print('An innocent quokka is printing schema...')
    interaction001.printSchema()
#     root
#  |-- user_id: string (nullable = true)
#  |-- count: long (nullable = true)
#  |-- track_id: string (nullable = true)
#  |-- __index_level_0__: long (nullable = true)
    print('Total number of obs')
    test = spark.sql("SELECT count(*) FROM interaction001")
    test.show()
    print('First 10 lines')
    test2 = spark.sql("SELECT * FROM interaction001 LIMIT 10")
    test2.show()


    
    
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('final_project').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    
    infile = sys.argv[-1]
    # Call our main routine
    main(spark, netID, infile)
    
