import tensorflow as tf
import numpy as np
from network import *

def ReadInput(filename):
  FirstLine = True
  Titles = []
  Data = []
  with open(filename) as f:
    for lines in f:
      line = lines.strip().split(",")
      if FirstLine:
        FirstLine = False
        Titiles = line
        continue
      Data.append([float(v) for v in line])
  
  Data = np.array(Data)

  return Titles, Data


if __name__ == "__main__":
  filename = "data/train.csv"
  Titles, Dataset = ReadInput(filename)
  savepath = "./model/model"
  RegCoeff = 0.0
  NumEpoch = 200
  NumBatch = 256
  ActType = "Relu"

  # Build Network
  inputs = tf.placeholder(tf.float32, shape=[None, Dataset.shape[1]-1])
  labels = tf.placeholder(tf.float32, shape=[None])

  layers, weights, bnvars = BuildNetwork(inputs, True, ActType)
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
  for epoch in range(NumEpoch):
    np.random.shuffle(Dataset)
    X = Dataset[:,:-1]
    Y = Dataset[:,-1]
    for i in range(X.shape[0]//NumBatch+1):
      IndexMax = min(X.shape[0]-i*NumBatch, NumBatch)
      _, TrainLoss, MSELoss, RegLoss, _ = sess.run([TrainOp, loss, LossMSE, LossReg, bnvars], 
                                                feed_dict={inputs: X[i*NumBatch: i*NumBatch+IndexMax], 
                                                           labels: Y[i*NumBatch: i*NumBatch+IndexMax]})
      print "-----------------"
      print i, "/", X.shape[0]//NumBatch+1
      print "Total Loss: ", TrainLoss
      print "MSE Loss: ", MSELoss
      print "Reg Loss: ", RegLoss
    saver.save(sess, savepath)
