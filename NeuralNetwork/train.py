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
  filename = "data/train.csv"
  Dataset = ReadInput(filename)
  savepath = "./model/model_R0_B256_A0.9_L5"
  RegCoeff = 0.0
  NumEpoch = 100
  NumBatch = 256
  path = "./board/L5"

  # Build Network
  inputs = tf.placeholder(tf.float32, shape=[None, Dataset.shape[1]-1])
  labels = tf.placeholder(tf.float32, shape=[None])

  layers, weights, bnvars = BuildNetwork(inputs, True)
  saver = tf.train.Saver()
  
  output = layers[-1][:,0]

  # Define Loss & Optimizer
  LossMSE = tf.losses.mean_squared_error(labels, output)
  LossReg = RegCoeff*sum([tf.nn.l2_loss(w) for w in weights])
  loss = LossMSE + LossReg
  opt = tf.train.AdamOptimizer()
  
  TrainOp = opt.minimize(loss)

  # Begin Training
  sess = tf.Session()
  sess.run(tf.initializers.global_variables())
  record = []

  for epoch in range(NumEpoch):
    np.random.shuffle(Dataset)
    X = Dataset[:,:-1]
    Y = Dataset[:,-1]
    for i in range((X.shape[0]-1)//NumBatch+1):
      IndexMax = min(X.shape[0]-i*NumBatch, NumBatch)
      _, TrainLoss, MSELoss, RegLoss, _ = sess.run([TrainOp, loss, LossMSE, LossReg, bnvars], 
                                                feed_dict={inputs: X[i*NumBatch: i*NumBatch+IndexMax], 
                                                           labels: Y[i*NumBatch: i*NumBatch+IndexMax]})
      record.append(TrainLoss)
      print "-----------------"
      print epoch, "/", NumEpoch
      print i, "/", X.shape[0]//NumBatch+1
      print "Total Loss: ", TrainLoss
      print "MSE Loss: ", MSELoss
      print "Reg Loss: ", RegLoss
    saver.save(sess, savepath)
  
  with open("./log.txt", "a") as f:
      for rec in record:
        f.write(str(rec) + "\n")
