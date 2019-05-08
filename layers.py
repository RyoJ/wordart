import numpy as np
from function import sigmoid, softmax

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):#重みから行ベクトルを抜き出す
        W, = self.params
        self.idx = idx
        out = W[idx]
#        if ch07.flag.get_flag(4)==False:
            #print('Embedding.out:{}'.format(out.shape)) 
            #print('Embedding.out:{}'.format(out)) 
            #何回くりかえすか知りたい       
        return out

class LSTM:
    def __init__(self, Wx, Wh, b):
        '''

        Parameters
        ----------
        Wx: 入力`x`用の重みパラーメタ（4つ分の重みをまとめる）
        Wh: 隠れ状態`h`用の重みパラメータ（4つ分の重みをまとめる）
        b: バイアス（4つ分のバイアスをまとめる）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        #if ch07.flag.get_flag(6)==False:
         #   print('LSTM.in.hprev:{}'.format(h_prev.shape)) 
            #print('LSTM.in.hprev:{}'.format(h_prev)) 
          #  print('LSTM.in.x:{}'.format(x.shape)) 
            #print('LSTM.in.x:{}'.format(x)) #入力
           # print('LSTM.in.b:{}'.format(b.shape)) 
            #print('LSTM.out.A:{}'.format(A)) 
            #print('LSTM.out.A:{}'.format(A.shape)) 
            #print('LSTM.out.A:{}'.format(A)) 

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        #if ch07.flag.get_flag(11)==False:
         #   print('LSTM.out.hnext:{}'.format(h_next.shape)) 
          #  print('LSTM.out.hnext:{}'.format(h_next)) 
           # print('LSTM.out.cnext:{}'.format(c_next.shape)) 
            #print('LSTM.out.cnext:{}'.format(c_next)) 
        return h_next, c_next

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out