import tensorflow as tf
import numpy as np
from network import *

def ReadInput(filename):
  Data = []
  with open(filename) as f:
    for lines in f:
      line = lines.strip().split(",")
      Data.append([float(v) for v in line])
  
  Data = np.array(Data)

  return Data


if __name__ == "__main__":
  testfilename = "data/test.csv"
  TestData = ReadInput(testfilename)
  trainfilename = "data/train.csv"
  TrainData = ReadInput(trainfilename)
  savepath = "./model/model_R0_B256_A0.9_L5"
  NumBatch = 256

  # Build Network
  inputs = tf.placeholder(tf.float32, shape=[None, TrainData.shape[1]-1])
  labels = tf.placeholder(tf.float32, shape=[None])

  layers, weights, bnvars = BuildNetwork(inputs, False)
  saver = tf.train.Saver()
  
  output = layers[-1][:,0]

  # Define Loss & Optimizer
  SqLoss = tf.reduce_sum((labels-output)**2)
  
  # Begin Training
  sess = tf.Session()
  saver.restore(sess, savepath)
  
  # Test Error
  X_test = TestData[:,:-1]
  Y_test = TestData[:,-1]
  CumError = 0.0
  for i in range(X_test.shape[0]//NumBatch+1):
    IndexMax = min(X_test.shape[0]-i*NumBatch, NumBatch)
    SqureError = sess.run([SqLoss], 
                          feed_dict={inputs: X_test[i*NumBatch: i*NumBatch+IndexMax], 
                                     labels: Y_test[i*NumBatch: i*NumBatch+IndexMax]})
    CumError += SqureError[0]
  TestRMSE = np.sqrt(CumError / X_test.shape[0])
  print "Test RMSE: ", TestRMSE
  
  # Train Error
  X_train = TrainData[:,:-1]
  Y_train = TrainData[:,-1]
  CumError = 0.0
  for i in range(X_train.shape[0]//NumBatch+1):
    IndexMax = min(X_train.shape[0]-i*NumBatch, NumBatch)
    SqureError = sess.run([SqLoss], 
                          feed_dict={inputs: X_train[i*NumBatch: i*NumBatch+IndexMax], 
                                     labels: Y_train[i*NumBatch: i*NumBatch+IndexMax]})
    CumError += SqureError[0]
  TrainRMSE = np.sqrt(CumError / X_train.shape[0])
  print "Train RMSE: ", TrainRMSE
