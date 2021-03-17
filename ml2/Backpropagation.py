import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target



def relu(x):
    return np.maximum(0.0,x)


def softmax(x):
    maxVal=np.amax(x)
    x=x-maxVal
    sigma=np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
    return sigma

def computeLayer(X, W, b):
    return np.matmul(X,W)+b


def CE(target, prediction,dataShape):
    averageCE=-np.sum(target * np.log(prediction))/dataShape
    return averageCE

def gradCE(target, prediction):
    return (prediction - target)/N

def der_relu(x):
    x[x > 0] = 1
    x[x < 0] = 0
    return x

# the gradient of the loss with respect to the output layer weights
def gradLossW_o(target, prediction,hiddenLayer):
    CE=gradCE(target,prediction)
    return np.matmul(np.transpose(hiddenLayer),CE)


# the gradient of the loss with respect to the output layer bias
def gradLossb_o(target, prediction,hiddenLayer):
    ones=np.ones((1,N))
    CE=np.matmul(ones,gradCE(target,prediction))
    return CE

# the gradient of the loss with respect to the hidden layer weights
def gradLossW_h(target,prediction,hiddenLayer,x_in,W_out): # x_in is train_data
    CE=gradCE(target,prediction)
    loss_W_h=np.matmul(np.transpose(x_in),der_relu(hiddenLayer)*np.matmul(CE,np.transpose(W_out)))
    return loss_W_h

# the gradient of the loss with respect to the hidden layer biases
def gradLossb_h(target,prediction,hiddenLayer,x_in,W_out):
    CE=gradCE(target,prediction)
    loss_W_h=np.matmul(CE,np.transpose(W_out))*der_relu(hiddenLayer)
    ones=np.ones((1,N))
    return  np.matmul(ones,loss_W_h)




def forward_prop(x, target, W_o, b_o, W_h, b_h):
    z = computeLayer(x, W_h, b_h)
    h = relu(z)
    o = computeLayer(h, W_o, b_o)
    p = softmax(o)

    pred = np.argmax(p, axis=1)
    target = np.argmax(target, axis=1)

    acc = np.sum(pred == target) / x.shape[0]

    return z, h, o, p, acc



def init(unitSize_h):
    mu=0
    W_o = np.random.normal(mu, np.sqrt(2 / (unitSize_h + 10)), (unitSize_h, 10))
    W_h = np.random.normal(mu, np.sqrt(2 / (trainData.shape[1] + unitSize_h)), (trainData.shape[1], unitSize_h))

    b_o = np.zeros((1, 10))
    b_h = np.zeros((1, unitSize_h))

    V_o = np.full((unitSize_h, 10), 1e-5)
    V_h = np.full((trainData.shape[1], unitSize_h), 1e-5)

    return W_o, W_h, b_o, b_h, V_o, V_h


def update(gamma, alpha, gW_o, gW_h, gb_o, gb_h, W_o, W_h, b_o, b_h, v_Wo, v_Wh):
    v_bo=b_o
    v_bh=b_h
    v_Wo = gamma * v_Wo + alpha * gW_o
    W_o = W_o - v_Wo
    v_bo = gamma * v_bo + alpha * gb_o
    b_o = b_o - v_bo

    v_Wh = gamma * v_Wh + alpha * gW_h
    W_h = W_h - v_Wh
    v_bh = gamma * v_bh + alpha * gb_h
    b_h = b_h - v_bh

    return W_o,W_h,b_o,b_h


def backProp(epochs,gamma, alpha,trainData, trainTarget_hot, validData, validTarget_hot, testData, testTarget_hot):

    W_o, W_h, b_o, b_h, v_Wo, v_Wh=init(unitSize_h=1000)

    train_acc_list = []
    valid_acc_list = []
    test_acc_list = []
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    i=0
    for i in range(epochs):
        
        z_h, hiddenLayer, z_o, Outlayer, train_acc= forward_prop(trainData,trainTarget_hot,W_o,b_o,W_h,b_h)
        z_h2, hiddenLayer2, z_o2, Outlayer2, valid_acc= forward_prop(validData,validTarget_hot,W_o,b_o,W_h,b_h)
        z_h3, hiddenLayer3, z_o3, Outlayer3, test_acc= forward_prop(testData,testTarget_hot,W_o,b_o,W_h,b_h)
        prediction=Outlayer

        trainLoss=CE(trainTarget_hot,prediction,trainData.shape[0])
        validLoss=CE(validTarget_hot,Outlayer2,validData.shape[0])
        testLoss=CE(testTarget_hot,Outlayer3,testData.shape[0])

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(trainLoss)
        valid_loss_list.append(validLoss)
        test_loss_list.append(testLoss)
        
        # Calculating all gradients
        gW_o=gradLossW_o(trainTarget_hot, prediction,hiddenLayer)
        gb_o=gradLossb_o(trainTarget_hot, prediction,hiddenLayer)
        gW_h=gradLossW_h(trainTarget_hot,prediction,hiddenLayer,trainData,W_o)
        gb_h=gradLossb_h(trainTarget_hot,prediction,hiddenLayer,trainData,W_o)

        W_o, W_h, b_o, b_h=update(gamma,alpha,gW_o,gW_h,gb_o,gb_h,W_o, W_h, b_o, b_h, v_Wo, v_Wh)
        
    return train_acc_list,test_acc_list,valid_acc_list,train_loss_list,valid_loss_list, test_loss_list



def plot(figureNum, title,yLabel,trainArray,validArray):  
    f = plt.figure(figureNum)
    title = title 
    plt.title(title)  
    plt.ylabel(yLabel)
    plt.xlabel('Iterations')   
    plt.plot(trainArray)
    plt.plot(validArray)  
    plt.legend(["Training "+yLabel,"Valid "+yLabel],loc='lower right')
    # plt.show()
    plt.savefig(str(figureNum))


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget_hot, validTarget_hot, testTarget_hot = convertOneHot(trainTarget,validTarget,testTarget)

trainData=trainData.reshape(10000,784)
validData=validData.reshape(6000,784)
testData=testData.reshape(2724,784)
N=trainData.shape[0]


train_acc_list,test_acc_list,valid_acc_list,train_loss_list,valid_loss_list, test_loss_list=backProp(200,0.9,0.1,trainData,trainTarget_hot,validData,validTarget_hot,testData,testTarget_hot)


plot(1, "Training and Validation Accuracy over 200 epochs, α=0.1, γ=0.9","Accuracy",train_acc_list,valid_acc_list)
# plot(2, "Training and Validation Loss over 200 epochs, α=0.1,  γ=0.9","Loss",train_loss_list,valid_loss_list)
