import tensorflow as tf
import numpy as np

def ActLayer(inputs, ActType):
  if ActType == "Relu":
    output = tf.nn.relu(inputs)
  elif ActType == "Sigmoid":
    output = tf.nn.sigmoid(inputs)
  elif ActType == "Tanh":
    output = tf.nn.tanh(inputs)

  return output


def BNLayer(inputs, IsTrain):
  alpha = 0.8
  Shape = inputs.get_shape().as_list()[1]
  MeanVar = tf.Variable(tf.zeros(Shape))
  ScaleVar = tf.Variable(tf.zeros(Shape))
  
  if IsTrain:
    # Batch Mean & Scale
    mean, scale = tf.nn.moments(inputs, axes=[0])
    
    # Compute and update moving average
    MovingMean = alpha*MeanVar + (1-alpha)*mean
    MovingScale = alpha*ScaleVar + (1-alpha)*scale

    MeanVar = tf.assign(MeanVar, MovingMean, False, False)
    ScaleVar = tf.assign(ScaleVar, MovingScale, False, False)

    # Apply Batch Normalization
    output = tf.nn.batch_normalization(inputs, mean, scale, None, None, 1e-20)
  else:
    output = tf.nn.batch_normalization(inputs, MeanVar, ScaleVar, None, None, 1e-20)
  
  return output, [MeanVar, ScaleVar]


def MatmulLayer(inputs, NumNeuron):
  Shape = inputs.get_shape().as_list()[1]
  
  W = tf.Variable(tf.random_normal([Shape, NumNeuron], stddev=0.1))
  bias = tf.Variable(tf.random_normal([NumNeuron], stddev=0.1))

  output = tf.matmul(inputs, W) + bias

  return output, W


def BuildNetwork(inputs, IsTrain, ActType):
  layers = []
  weights = []
  bnvars = []

  # Preprocess Input
  layer, bnvar = BNLayer(inputs, IsTrain)
  layers.append(layer)
  bnvars.append(bnvar)

  layers.append(ActLayer(layers[-1], ActType))
  
  # First Layer
  layer, weight = MatmulLayer(layers[-1], 1000)
  layers.append(layer)
  weights.append(weight)

  layer, bnvar = BNLayer(layers[-1], IsTrain)
  layers.append(layer)
  bnvars.append(bnvar)

  layers.append(ActLayer(layers[-1], ActType))
  
  # Second Layer
  layer, weight = MatmulLayer(layers[-1], 2000)
  layers.append(layer)
  weights.append(weight)
  
  layer, bnvar = BNLayer(layers[-1], IsTrain)
  layers.append(layer)
  bnvars.append(bnvar)

  layers.append(ActLayer(layers[-1], ActType))
  
  # Third Layer
  layer, weight = MatmulLayer(layers[-1], 3000)
  layers.append(layer)
  weights.append(weight)
  
  layer, bnvar = BNLayer(layers[-1], IsTrain)
  layers.append(layer)
  bnvars.append(bnvar)

  layers.append(ActLayer(layers[-1], ActType))
  
  # Output Layer
  layer, weight = MatmulLayer(layers[-1], 1)
  layers.append(layer)
  weights.append(weight)
  
  return layers, weights, bnvars
