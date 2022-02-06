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



def main(spark, netID):
    cf = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_train_new.parquet') 
    cf_test = spark.read.parquet(f'hdfs:/user/bm106/pub/MSD/cf_validation.parquet')
    
    cf.createOrReplaceTempView('cf')
    cf_test.createOrReplaceTempView('cf_test')
    
    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_Index")
    indexed_user = indexer_user.setHandleInvalid("skip").fit(cf)
    cf = indexed_user.transform(cf)
    cf_test = indexed_user.transform(cf_test)   
    
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_Index")
    indexed_track = indexer_track.setHandleInvalid("skip").fit(cf)
    cf = indexed_track.transform(cf)
    cf_test = indexed_track.transform(cf_test) 
    
    ###From our train_val results, we obtain the best set of hyperparameters as Rank=20, regParam=0.01
    rank = 90
    regParam = 0.01
    scale = 40
    max_iter = 20
    infile = 'cf_train_new'
    als = ALS(rank=rank, maxIter=max_iter,
                regParam=regParam, alpha = scale, userCol="user_Index", 
                itemCol="track_Index", ratingCol="count", implicitPrefs=True,
                nonnegative=True,coldStartStrategy="drop")
    model = als.fit(cf)
    als.write().overwrite().save(f'model/als_{rank}_{regParam}_{max_iter}_{scale}_{infile}')
    model.write().overwrite().save(f'model/model_{rank}_{regParam}_{max_iter}_{scale}_{infile}')
    print('model saved')
    
    
    # als = ALS.load(f'model/als_{rank}_{regParam}_cf_train')
    # model = ALSModel.load(f'model/model_{rank}_{regParam}_cf_train')


    user_ids = cf_test.select(als.getUserCol()).distinct()
    recoms = model.recommendForUserSubset(user_ids,500)
    predictions = recoms.select('user_Index','recommendations.track_Index')
    print('predictions done')
    
    ### Group by user index and aggregate
    truth = cf_test.select('user_Index', 'track_Index').groupBy('user_Index').agg(F.expr('collect_list(track_Index) as truth'))
    print('truth done')
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

    main(spark, netID)