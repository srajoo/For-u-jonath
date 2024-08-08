import sys
import os
from pyspark import SparkContext, StorageLevel, SparkConf
import time
import json
import numpy as np
import xgboost as xgb
import csv


os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'


def writeToCSV(output_file_name, predictionList):
    with open(output_file_name, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(["user_id", "business_id", "prediction"])
        for row in predictionList:
            writer.writerow(row)


def main():

    #folder_path = sys.argv[1]
    #test_file_name = sys.argv[2]
    #output_file_name = sys.argv[3]

    folder_path = "./"
    test_file_name = "yelp_val_in.csv"
    output_file_name = "output2_2.csv"

    configuration = SparkConf()
    configuration.set("spark.driver.memory", "4g")
    configuration.set("spark.executor.memory", "4g")
    sc = SparkContext.getOrCreate(configuration)
    sc.setLogLevel("ERROR")

    start = time.time()

    rdd = sc.textFile(os.path.join(folder_path, 'yelp_train.csv'))
    header = rdd.first()
    trainRDD = rdd.filter(lambda line: line != header) \
        .map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))

    rdd = sc.textFile(test_file_name)
    header = rdd.first()
    testRDD = rdd.filter(lambda line: line != header).map(lambda x: (x.split(',')[0], x.split(',')[1]))

    userFeatures = sc.textFile(os.path.join(folder_path, 'user.json')).map(lambda x: json.loads(x))\
        .map(lambda x: (x['user_id'], (int(x['review_count']), float(x['average_stars'])))).collectAsMap()

    businessFeatures = sc.textFile(os.path.join(folder_path, 'business.json')).map(lambda x: json.loads(x))\
        .map(lambda x: (x['business_id'], (int(x['review_count']), float(x['stars'])))).collectAsMap()

    trainUtilityMatrix = trainRDD.map(lambda x: (x[0], x[1], userFeatures.get(x[0]), businessFeatures.get(x[1]), x[2]))\
        .map(lambda x: (x[0], x[1], x[2][0], x[2][1], x[3][0], x[3][1], x[4])).collect()

    testUtilityMatrix = testRDD.map(lambda x: (x[0], x[1], userFeatures.get(x[0]), businessFeatures.get(x[1]))) \
        .map(lambda x: (x[0], x[1], x[2][0], x[2][1], x[3][0], x[3][1])).collect()

    arrayTrain = np.array(trainUtilityMatrix)
    arrayTest = np.array(testUtilityMatrix)

    X_train = np.array(arrayTrain[:, [2,3,4,5]])
    y_train = np.array(arrayTrain[:,6], dtype="float")

    X_test = np.array(arrayTest[:, [2,3,4,5]])

    model = xgb.XGBRegressor(objective='reg:linear')
    model.fit(X_train,y_train)

    predictions = list(model.predict(X_test))
    predictionList = zip(zip(*testRDD.collect()), predictions)

    writeToCSV(output_file_name, predictionList)

    end = time.time()
    print("Duration:", end-start)


if __name__ == '__main__':
    main()