import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        y = x - np.max(x)
        z = np.exp(y) / np.sum(np.exp(y))
        #if fl.pas(20):
         #   print('softmax.x:{}'.format(x.shape))
          #  print('softmax.x:{}'.format(x))
           # print('softmax.y:{}'.format(y.shape))
            #print('softmax.y:{}'.format(y))
            #print('softmax.z=p:{}'.format(z.shape))
            #print('softmax.z=p:{}'.format(z)) 
        x=z           
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))