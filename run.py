# ----------------------------------------------------------------------
# Copyright (c) 2017, Jin-Man Park. All rights reserved.
# Contributors: Jin-Man Park and Jong-hwan Kim
# Affiliation: Robot Intelligence Technology Lab.(RITL), Korea Advanced Institute of Science and Technology (KAIST)
# URL: http://rit.kaist.ac.kr
# E-mail: jmpark@rit.kaist.ac.kr
# Citation: Jin-Man Park, and Jong-Hwan Kim. "Online recurrent extreme learning machine and its application to
# time-series prediction." Neural Networks (IJCNN), 2017 International Joint Conference on. IEEE, 2017.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
# ----------------------------------------------------------------------
# This code is originally from Numenta's Hierarchical Temporal Memory (HTM) code
# (Numenta Platform for Intelligent Computing (NuPIC))
# And modified to run Online Recurrent Extreme Learning Machine (OR-ELM)
# ----------------------------------------------------------------------
import csv
from optparse import OptionParser
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import pandas as pd
from errorMetrics import *
from OR_ELM import ORELM
from FOS_ELM import FOSELM
import time
import glob


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#from algorithms.NAOS_ELM import NAOSELM

target_col = 's1_o3'


def _getArgs():
  parser = OptionParser(usage="%prog [options]"
                              "\n\nOnline Recurrent Extreme Learning Machine (OR-ELM)"
                              "and its application to time-series prediction,"
                              "with NYC taxi passenger dataset.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='spain',
                    dest="dataSet",
                    help="DataSet Name, choose from sine or nyc_taxi")
  parser.add_option("-l",
                    "--numLags",
                    type=int,
                    default='100',
                    help="the length of time window, this is used as the input dimension of the network")
  parser.add_option("-p",
                    "--predStep",
                    type=int,
                    default='1',
                    help="the prediction step of the output")
  parser.add_option("-a",
                    "--algorithm",
                    type=str,
                    default='ORELM',
                    help="Algorithm name, choose from FOSELM, NFOSELM, NAOSELM, ORELM")
  (options, remainder) = parser.parse_args()
  return options, remainder


def initializeNet(nDimInput, nDimOutput, numNeurons=100, algorithm='ORELM',
                  LN=True, InWeightFF=0.999, OutWeightFF=0.999, HiddenWeightFF=0.999,
                  ORTH=True, AE=True, PRINTING=True):

  assert algorithm =='FOSELM' or algorithm == 'NFOSELM' or algorithm == 'NAOSELM' or algorithm == 'ORELM'
  if algorithm=='FOSELM':
      '''
      Fully Online Sequential ELM (FOSELM). It's just like the basic OSELM, except its initialization.
      Wong, Pak Kin, et al. "Adaptive control using fully online sequential-extreme learning machine
      and a case study on engine air-fuel ratio regulation." Mathematical Problems in Engineering 2014 (2014).
      '''
      net = FOSELM(nDimInput, nDimOutput,
                  numHiddenNeurons=numNeurons,
                  activationFunction='sig',
                  forgettingFactor=OutWeightFF,
                  LN=False, ORTH=ORTH)
      '''             
  
      '''
  
  elif algorithm=='ORELM':
      '''
      Online Recurrent Extreme Learning Machine (OR-ELM).
      FOSELM + layer normalization + forgetting factor + input layer weight auto-encoding + hidden layer weight auto-encoding.
      '''
      net = ORELM(nDimInput, nDimOutput,
                  numHiddenNeurons=numNeurons,
                  activationFunction='sig',
                  LN=LN,
                  inputWeightForgettingFactor=InWeightFF,
                  outputWeightForgettingFactor=OutWeightFF,
                  hiddenWeightForgettingFactor=HiddenWeightFF,
                  ORTH=ORTH,
                  AE=AE)


  if PRINTING:
    print('----------Network Configuration-------------------')
    print('Algotirhm = '+algorithm)
    print('#input neuron = '+str(nDimInput))
    print('#output neuron = '+str(nDimOutput))
    print('#hidden neuron = '+str(numNeurons))
    print('Layer normalization = ' + str(net.LN))
    print('Orthogonalization = '+str(ORTH))
    print('Auto-encoding = '+str(AE))
    print('input weight forgetting factor = '+str(InWeightFF))
    print('output weight forgetting factor = ' + str(OutWeightFF))
    print('hidden weight forgetting factor = ' + str(HiddenWeightFF))
    print('---------------------------------------------------')

  return net

def readDataSet(dataSet):
  
  if dataSet=='spain':
    
    #df = pd.read_csv('./spain.csv' ,  sep = ';' , parse_dates=True)
    
    
    import glob
    filePath = 'data/'+dataSet+'.csv'
    files = glob.glob("./Data/*.csv")
    df = []
    for f in files:
        csv = pd.read_csv(f ,  sep = ';' , parse_dates=True)
        df.append(csv)
    df = pd.concat(df)
    
    df.rename(columns={'location ':'location'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']) 
    df = df.sort_values(by = 'date')
    df.set_index('date' , inplace=True)

    df.head()
    
    seq = df.copy()              

    
  else:
    raise(' unrecognized dataset type ')

  return seq



def getTimeEmbeddedMatrix(sequence, numLags, predictionStep):
  print ("generate time embedded matrix ")
  inDim = numLags
  X = np.zeros(shape=(len(sequence), inDim))
  T = np.zeros(shape=(len(sequence), 1))
  for i in range(numLags-1, len(sequence)-predictionStep):
    X[i, :] = np.array(sequence[target_col][(i-numLags+1):(i+1)])
    T[i, :] = sequence[target_col][i+predictionStep]
  return (X, T)



def saveResultToFile(dataSet, predictedInput, algorithmName,predictionStep):
  inputFileName = 'data/' + dataSet + '.csv'
  inputFile = open(inputFileName, "r")
  csvReader = csv.reader(inputFile)
  # skip header rows
  next(csvReader)
  #next(csvReader)
  #next(csvReader)
  outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'

  '''
  outputFile = open(outputFileName, "w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(
    ['timestamp', 'data', 'prediction-' + str(predictionStep) + 'step'])
  csvWriter.writerow(['datetime', 'float', 'float'])
  csvWriter.writerow(['', '', ''])

  for i in range(len(sequence)):
    row = next(csvReader)
    csvWriter.writerow([row[0], row[1], predictedInput[i]])

  outputFile.close()
  '''
  
  with open(outputFileName, 'w') as csvfile:
      csvwriter = csv.writer(csvfile) 
 
      csvwriter.writerow(['timestamp', 'data', 'prediction-' + str(predictionStep) + 'step'])    
      #csvwriter.writerows(['', '', ''])
      for i in range(len(sequence)):
          row = next(csvReader)
          row = row[0].split(';')
          csvwriter.writerow([row[0], row[1], predictedInput[i]])
  
  
  inputFile.close()
  print ('Prediction result is saved to ' + outputFileName)


if __name__ == "__main__":

  (_options, _args) = _getArgs()
  algorithm = _options.algorithm
  dataSet = _options.dataSet
  numLags = _options.numLags
  predictionStep = _options.predStep
  print ("run ", algorithm, " on ", dataSet)
  # prepare dataset
  sequence = readDataSet(dataSet)
  
  #sequence = sequence.head(50000)
  
  # standardize data by subtracting mean and dividing by std
  colss = sequence.columns
  encoder = LabelEncoder()
  sequence['location'] = encoder.fit_transform(sequence['location'])
  sequence['location'] = sequence['location']#.astype('float32')

  sequence.location.unique()
  
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_dataset = scaler.fit_transform(sequence)

  df1 = pd.DataFrame(scaled_dataset)
  df1.columns = colss

  df1.head()

  (X, T) = getTimeEmbeddedMatrix(df1, numLags, predictionStep)
  

  #n_train = int (len (X) * 50 / 100)

  #X_train, X_test = X[0:n_train,] , X[n_train:,]
  #print('X_train' ,X_train.shape)
  #print('X_test' ,X_test.shape)

  #Y_train, Y_test = T[0:n_train,] , T[n_train:,]
  #print('Y_train' ,Y_train.shape)
  #print('Y_test' ,Y_test.shape)

  random.seed(6)

  net = initializeNet(nDimInput=X.shape[1],
                      nDimOutput=1,
                      numNeurons=100,
                      algorithm=algorithm,
                      LN=True,
                      InWeightFF=0.995,
                      OutWeightFF=0.915,
                      HiddenWeightFF=0.955,
                      AE=True,
                      ORTH=True)
  net.initializePhase(lamb = 0.0001)

  predictedInput = np.zeros((len(df1),))
  targetInput = np.zeros((len(df1),))
  trueData = np.zeros((len(df1),))
  
  print('training started')
  
  t1 = time.time()
  for i in range(numLags, len(X)-predictionStep-1):
      net.train (X[[i], :] , T[[i], :])
      Y =  net.predict(X[[i], :])
      if Y[-1] < 0:
          predictedInput[i+1] = T[i] 
      elif Y[-1] > 1:
          predictedInput[i+1] = T[i]
      else:
          predictedInput[i+1] = Y[-1]
          
      targetInput[i+1] = df1[target_col][i+1+predictionStep]
      trueData[i+1] = df1[target_col][i+1]
      print ("{:5}th timeStep -  target: {:8.4f}   |    prediction: {:8.4f} ".format(i, targetInput[i+1], predictedInput[i+1]))
  t2 = time.time()
  
  print('training time: ', t2 - t1)
  print('training completed')
     
  
    
  # Reconstruct original value

  r1 = np.concatenate((predictedInput.reshape(-1,1),sequence.values[:,1:]), axis =1)
  rs1 = scaler.inverse_transform(r1)
  predictedInput = rs1[:,0:1]
  
  
  r2 = np.concatenate((targetInput.reshape(-1,1),sequence.values[:,1:]), axis =1)
  rs2 = scaler.inverse_transform(r2)
  targetInput = rs2[:,0:1]

  r3 = np.concatenate((trueData.reshape(-1,1),sequence.values[:,1:]), axis =1)
  rs3 = scaler.inverse_transform(r3)
  trueData = rs3[:,0:1]
  
  # Calculate NRMSE from stpTrain to the end
  skipTrain = numLags
  from plot import computeSquareDeviation

  squareDeviation = computeSquareDeviation(predictedInput, targetInput)
  squareDeviation[:skipTrain] = None
  nrmse = np.sqrt(np.nanmean(squareDeviation)) #/ np.nanstd(targetInput)
  print ("NRMSE {}".format(nrmse))
  # Save prediction result as csv file
  #saveResultToFile(dataSet, predictedInput, 'FF' + str(net.forgettingFactor) + algorithm + str(net.numHiddenNeurons), predictionStep)

  '''
  #Plot predictions and target values
  
'''
  
  from numpy import *
  import math
  import matplotlib.pyplot as plt

  t = linspace(0, 2*math.pi, 7954)
  a = sin(t)
  b = cos(t)
  c = a + b

  plt.plot(t, a, 'r') # plotting t, a separately 
  plt.plot(t, b, 'b') # plotting t, b separately 
  plt.plot(t, c, 'g') # plotting t, c separately 
  plt.show()
    
  print ("predicted shape: " , predictedInput.shape)
  print ("target shape: " , targetInput.shape)
  
  plt.figure(figsize=(15,6))
  targetPlot, = plt.plot(targetInput[skipTrain:,:],label = 'target' , color='r'  ,marker='.',linestyle='-')
  predictedPlot, = plt.plot(  predictedInput[skipTrain:, :], label = 'Predicted' ,color='b'  ,marker='.',linestyle=':')
  print('Drawned')
  #plt.xlim([0,8000])
  #plt.ylim([0, 1030])
  plt.ylabel('value',fontsize=15)
  plt.xlabel('time',fontsize=15)
  plt.ion()
  plt.grid()
  plt.legend(handles=[targetPlot, predictedPlot])
  plt.title('Time-series Prediction of '+algorithm+' on '+dataSet+' dataset',fontsize=20,fontweight=40)
  plot_path = './fig/predictionPlot.png'
  fname = "myfile.png"
  plt.tight_layout()
  plt.savefig( fname, dpi=100)
  plt.draw()
  plt.show()
  plt.pause(0)
  print ('Prediction plot is saved to'+plot_path )
  #from plotResults import load_plot_return_errors
    
  algorithm= 'OR_ELM'
  plt.figure(figsize=(15,6))
  targetPlot,=plt.plot(np.abs(targetInput[skipTrain: , :]-targetInput[skipTrain: , :]),label='target',color='red',marker='.',linestyle='-')
  plt.xlim([5500,6500])
  #plt.ylim([0, 30000])
  plt.ylabel('absolute error',fontsize=15)
  plt.xlabel('time',fontsize=15)
  plt.ion()
  plt.grid()
  plt.legend(handles=[targetPlot, predictedPlot])
  plt.title('Prediction error of '+algorithm+' on '+dataSet+' dataset',fontsize=20,fontweight=40)
  plot_path = './predictionErrorPlot.png'
  fname1 = "myfileError.png"
  plt.savefig( fname1, dpi=100)  
  plt.draw()
  plt.show()
  plt.pause(0)
  print('Prediction error plot is saved to'+plot_path)
  
  '''
  plt.figure(figsize=(15,6))
  targetPlot,=plt.plot(targetInput,label='target',color='red',marker='.',linestyle='-')
  predictedPlot,=plt.plot(predictedInput,label='predicted',color='blue',marker='.',linestyle=':')
  plt.xlim([13000,13500])
  plt.ylim([0, 30000])
  plt.ylabel('value',fontsize=15)
  plt.xlabel('time',fontsize=15)
  plt.ion()
  plt.grid()
  plt.legend(handles=[targetPlot, predictedPlot])
  plt.title('Time-series Prediction of '+algorithm+' on '+dataSet+' dataset',fontsize=20,fontweight=40)
  plot_path = './fig/predictionPlot.png'
  plt.savefig(plot_pathbbox_inches='tight' , fname)
  plt.draw()
  plt.show()
  plt.pause(0)
  print ('Prediction plot is saved to'+plot_path)
  '''