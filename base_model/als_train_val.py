#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getpass
import sys
import numpy as np
import pyspark.sql.functions as F
from random import sample
from pyspark import SparkConf
from pyspark.sql import SparkSession, functions
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALS,ALSModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import subprocess



def main(spark, netID, infile):
    if infile == 'cf_train':
        cf = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train.parquet')
    else:
        ###Test on a subset of training data
        cf = spark.read.parquet(f'hdfs://horton.hpc.nyu.edu:8020/user/yd1008/{infile}.parquet')
    cf_val = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_validation.parquet')  
    #cf_test = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_test.parquet')  
    cf.createOrReplaceTempView('cf')
    cf_val.createOrReplaceTempView('cf_val')   
    #cf_test.createOrReplaceTempView('cf_test')  

    ### Select users
    train_users = set(row['user_id'] for row in cf.select('user_id').distinct().collect())  
    val_users = set(row['user_id'] for row in cf_val.select('user_id').distinct().collect())
    print(f'Number of users in training: {len(train_users)}')
    print(f'Number of users in validation: {len(val_users)}')
    ### Get distinct users in training
    distinct_users = list(train_users-val_users)
    print(f'Number of distinct users in training: {len(distinct_users)}')
    ### Downsample training data
    sub_distinct_users = sample(distinct_users,int(len(distinct_users)*0.25))
    print(f'Number of distinct users after downsample: {len(sub_distinct_users)}')
    downsampled_users = [x for x in sub_distinct_users+list(val_users)]
    print(f'Total number of train users after downsample: {len(downsampled_users)}')   
    ### Filter train data by randomly omitting 75% of non-overlapping users
    cf = cf.where(F.col('user_id').isin(downsampled_users))
    cf.createOrReplaceTempView('cf')
    ### Debug
    #count_reduced = len(spark.sql('SELECT DISTINCT user_id FROM cf').collect())
    #print(f"After subsetting, there are {count_reduced} users in training")
    ### Create indexers to convert strings to doubles
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_Index")
    indexed_user = indexer_user.setHandleInvalid("skip").fit(cf)
    cf = indexed_user.transform(cf)
    cf_val = indexed_user.transform(cf_val)   
    #cf_test = indexed_user.transform(cf_test)   
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_Index")
    indexed_track = indexer_track.setHandleInvalid("skip").fit(cf)
    cf = indexed_track.transform(cf)
    cf_val = indexed_track.transform(cf_val)
    #cf_test = indexed_track.transform(cf_test)   

    ### See if there exists a trained model 
    def run_cmd(args_list):
        proc = subprocess.Popen(args_list, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        proc.communicate()
        return proc.returncode
    

    ###Create model
    ranks = [5,10,15,20]
    regParams = [1.,0.1,0.01]
    for rank in ranks:
        for regParam in regParams:

            cmd = ['hdfs', 'dfs', '-test', '-d', f"model/model_{rank}_{regParam}_{infile}"]
            code = run_cmd(cmd)
            if code == 0:
                print(f"/model/model_{rank}_{regParam}_{infile} exist \n Loading model...")
                print(f'code:{code}') 
                als = ALS.load('model/als_cf_train')
                model = ALSModel.load('model/model_cf_train')
            else:  
                print(f'code:{code}') 
                print('No trained model found, start training...')
                als = ALS(rank=rank, maxIter=5, seed=40, regParam=regParam, userCol="user_Index", itemCol="track_Index", ratingCol="count", implicitPrefs=True,nonnegative=True,coldStartStrategy="drop")
                model = als.fit(cf)
                ### Save model for testing set
                als.write().overwrite().save(f'model/als_{rank}_{regParam}_{infile}')
                model.write().overwrite().save(f'model/model_{rank}_{regParam}_{infile}')
                print('model saved')
            user_ids = cf_val.select(als.getUserCol()).distinct()
            recoms = model.recommendForUserSubset(user_ids,500)
            predictions = recoms.select('user_Index','recommendations.track_Index')
            print('predictions done')
            ### Group by user index and aggregate
            truth = cf_val.select('user_Index', 'track_Index').groupBy('user_Index').agg(F.expr('collect_list(track_Index) as truth'))
            print('truth done')
            ###DEBUG
            ###print(f'# rows in truth is: {truth.count()}')
            ### join prediction and truth. rdd is required here 
            combined = predictions.join(functions.broadcast(truth), 'user_Index', 'inner').rdd
            combined_mapped = combined.map(lambda x: (x[1], x[2]))
            print('rdd created')
            ### Metrics ref:https://spark.apache.org/docs/2.3.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics
            metrics = RankingMetrics(combined_mapped)
            print('metrics done')
            ### Mean Average Precision
            MAP = metrics.meanAveragePrecision
            ### Normalized Discounted Cumulative Gain
            ndcg = metrics.ndcgAt(500)
            ### Precision at k
            pk = metrics.precisionAt(500)
            print(f'Rank:{rank}, regParam: {regParam}, map score:  {MAP}, ndcg score: {ndcg}, pk score: {pk}')








    
if __name__ == "__main__":

#     conf = SparkConf()
#     conf.set("spark.executor.memory", "32G")
#     conf.set("spark.driver.memory", '32G')
#     conf.set("spark.driver.memoryOverhead", '16G')
#     conf.set("spark.executor.cores", "4")
#     conf.set("spark.sql.autoBroadcastJoinThreshold", '-1')
#     conf.set('spark.executor.instances','10')
#     conf.set('spark.yarn.driver.memoryOverhead','16G')
#     conf.set('spark.yarn.executor.memoryOverhead','16G')
#     conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
#     conf.set("spark.dynamicAllocation.enabled","true")
#     conf.set("spaek.yarn.am.memory","16G")
#     conf.set("spark.yarn.memoryOverhead","16G")
    #conf.set("spark.default.parallelism", "40")
    #spark.kryoserializer.buffer 
    spark = SparkSession.builder.appName('tuning').config('spark.blacklist.enabled', False).getOrCreate()
    
    #configurations = spark.sparkContext.getConf().getAll()
    #for conf in configurations:
    #    print(conf)
        
    netID = getpass.getuser()
    
    infile = sys.argv[-1]

    main(spark, netID, infile)