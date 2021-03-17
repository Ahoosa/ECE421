# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def loadData():
    with np.load('notMNIST.npz') as dataset:
        Data, Target = dataset['images'], dataset['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def y_hat(z):
    sigma = 1 / (1+np.exp(-z))
    return sigma


def loss(W, b, x, y, reg): 
    N = np.shape(y)[0]
    y_h = y_hat(np.matmul(x,W)+b)
    L_entropy = -np.sum(+y * np.log(y_h) +  (1-y)*np.log(1-y_h))  
    Loss = L_entropy/N + (reg/2) * (np.linalg.norm(W)**2)
    return Loss

def grad_loss(W, b, x, y, reg):
    y_h = y_hat(np.matmul(x,W)+b)
    gradCE = np.matmul(np.transpose(x),(y_h-y)) /np.shape(y)[0] 
    gradLoss_w = gradCE + reg*W
    gradLoss_b = np.sum(y_h-y)/np.shape(y)[0] 
    return gradLoss_w, gradLoss_b
    
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol):
    for i in range(epochs):
        grad_w,grad_b=grad_loss(W,b,x,y,reg)
        updated_w = W - alpha * grad_w
        updated_b = b - alpha * grad_b
        if np.linalg.norm(W-updated_w) < error_tol:
            return updated_w,updated_b
        else:
            W = updated_w
            b = updated_b
        
    return W,b



def testing_grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, validData, validTarget, testData, testTarget):

    training_loss = [loss(W, b, x, y, reg)]
    valid_loss = [loss(W, b, validData, validTarget, reg)]
    testing_loss = [loss(W, b, testData, testTarget, reg)]
    training_accuracy = [accuracy(W,x,b,y)]
    valid_accuracy =  [accuracy(W,validData,b,validTarget)] 
    testing_accuracy =  [accuracy(W,testData,b,testTarget)]

    for i in range(epochs):
        grad_w, grad_b  = grad_loss(W, b, x, y, reg)
        updated_w = W - alpha*grad_w
        updated_b = b - alpha*grad_b

        training_loss.append(loss(updated_w, updated_b, x, y, reg))
        valid_loss.append(loss(updated_w, updated_b, validData, validTarget, reg))
        testing_loss.append(loss(updated_w, updated_b, testData, testTarget, reg))
        training_accuracy.append(accuracy(updated_w,x,updated_b,y))
        valid_accuracy.append(accuracy(updated_w,validData,updated_b,validTarget))
        testing_accuracy.append(accuracy(updated_w,testData,updated_b,testTarget))   

        if np.linalg.norm(W-updated_w)<error_tol:

            return updated_w, updated_b, training_loss, training_accuracy, valid_loss, valid_accuracy, testing_loss, testing_accuracy

        else:
            W = updated_w
            b = updated_b

    return W, b, training_loss, training_accuracy, valid_loss, valid_accuracy, testing_loss, testing_accuracy


 

def accuracy(W,x,b,y):
    y_h = y_hat(np.matmul(x,W)+b)
    acc = np.sum((y_h>=0.5)==y)/np.shape(y)[0] 
    return acc


def plot(figureNum, title,yLabel,trainArray,validArray):  
    f = plt.figure(figureNum)
    title = title 
    plt.title(title)  
    plt.ylabel(yLabel)
    plt.xlabel('Iterations')   
    plt.plot(range(5001), trainArray, 'r', range(5001), validArray, 'b')  
    # plt.scatter(range(5001),trainArray)
    # plt.scatter(range(5001),validArray)
    plt.legend(["Training "+yLabel,"Valid "+yLabel],loc='upper right')
    plt.show()
    # plt.savefig(str(figureNum))


alpha = [0.005, 0.001, 0.0001] 
error_tol=0.0000001  
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

validData = validData.reshape((validData.shape[0],validData.shape[1]*validData.shape[2]))
trainData = trainData.reshape((trainData.shape[0],trainData.shape[1]*trainData.shape[2]))
testData = testData.reshape((testData.shape[0],testData.shape[1]*testData.shape[2]))
W = np.random.normal(0,0.5,(trainData.shape[1],1))
b=0
reg = [0.001, 0.1, 0.5]

# Changing the learning rate α with λ=0 and analyzing the plots 

 W, b, train_loss, train_accuracy, valid_loss, valid_accuracy, test_loss, test_accuracy = testing_grad_descent(W, b, trainData, trainTarget, alpha[0], 5000, 0, error_tol,validData, validTarget,testData,testTarget)
 plot(1, "Training and Validation Loss with α=0.005 & λ=0","Loss",train_loss,valid_loss)
 plot(2, "Training and Validation Accuracy with α=0.005 & λ=0","Accuracy",train_accuracy,valid_accuracy)


# Changing the regularization parameter λ with α=0.005 and analyzing the plots  

# W, b, train_loss, train_accuracy, valid_loss, valid_accuracy, test_loss, test_accuracy = testing_grad_descent(W, b, trainData, trainTarget, alpha[0], 5000, reg[0], error_tol,validData, validTarget,testData,testTarget)
# plot(7, "Training and Validation Loss with α=0.005 & λ=0.001","Loss",train_loss,valid_loss)
# plot(8, "Training and Validation Accuracy with α=0.005 & λ=0.001","Accuracy",train_accuracy,valid_accuracy)



#print(train_accuracy[5000])
#print(valid_accuracy[5000])
#print(test_accuracy[5000])




