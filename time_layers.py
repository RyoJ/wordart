import numpy as np
from layers import *

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs): #N：バッチサイズ、T：時系列データ数、V：語彙数、D：入力ベクトルの次元数
        N, T = xs.shape
        V, D = self.W.shape
        #if ch07.flag.get_flag(1)==False:
            #print('time_layers.TimeEmbedding.xs:{}'.format(xs))
            #print('time_layers.TimeEmbedding.N:{}'.format(N) + " " + 'T:{}'.format(T))
            #print('time_layers.TimeEmbedding.V:{}'.format(V) + " " + 'D:{}'.format(D))
            #print('time_layers.TimeEmbedding.W:{}'.format(self.W))

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
        #if ch07.flag.get_flag(5)==False:
         #   print('Embedding.TimeEm.in:{}'.format(xs.shape))
          #  print('Embedding.TimeEm.in:{}'.format(xs))
           # print('Embedding.TimeEm.out:{}'.format(out.shape))
            #print('Embedding.TimeEm.out:{}'.format(out))             
            self.layers.append(layer)

        return out

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        #if ch07.flag.get_flag(2)==False:
         #   print('time_layers.TimeLSTM.xs:{}'.format(xs.shape))#1,100
            #print('time_layers.TimeLSTM.xs:{}'.format(xs))
          #  print('time_layers.TimeLSTM.N:{}'.format(N) + " " + 'T:{}'.format(T))
           # print('time_layers.TimeLSTM.D:{}'.format(D))
            #print('time_layers.TimeLSTM.H:{}'.format(H))
            #print('time_layers.TimeLSTM.Wh:{}'.format(Wh.shape))
            #print('time_layers.TimeLSTM.Wh:{}'.format(Wh))

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
 
            self.layers.append(layer)
       # if ch07.flag.get_flag(15)==False:
        #    print('time_layers.TimeLSTM.in.xs:{}'.format(xs.shape))#1,100
            #print('time_layers.TimeLSTM.in.xs:{}'.format(xs)) 
         #   print('time_layers.TimeLSTM.self.h:{}'.format(self.h.shape))#1,100
          #  print('time_layers.TimeLSTM.h:{}'.format(self.h)) 
           # print('time_layers.TimeLSTM.self.c:{}'.format(self.c.shape))#1,100
            #print('time_layers.TimeLSTM.self.c:{}'.format(self.c)) 
            #print('time_layers.TimeLSTM.hs:{}'.format(hs))#1,100
            #print('time_layers.TimeLSTM.hs:{}'.format(hs.shape)) 
            #h,c,hsを表示する

            #print('time_layers.TimeLSTM.T:{}'.format(T))

        return hs

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params
        #if ch07.flag.get_flag(3)==False:
         #   print('time_layers.TimeAffine.N:{}'.format(N) + " " + 'T:{}'.format(T))
          #  print('time_layers.TimeAffine.D:{}'.format(D))
            #print('time_layers.TimeAffine.W:{}'.format(W))
           # print('time_layers.TimeAffine.W:{}'.format(W.shape))
            #print('time_layers.TimeAffine.b:{}'.format(b))
            #print('time_layers.TimeAffine.b:{}'.format(b.shape))

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        #if ch07.flag.get_flag(7)==False:
            #print('TimeAffine.in:{}'.format(rx))
         #   print('TimeAffine.in:{}'.format(rx.shape))
          #  print('TimeAffine.in:{}'.format(rx))
           # print('TimeAffine.W:{}'.format(W.shape))
            #print('TimeAffine.out:{}'.format(out))
            #print('TimeAffine.out:{}'.format(out.shape))
        self.x = x
        return out.reshape(N, T, -1)

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        #print('xs:{}'.format(xs.shape))
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        #print('ys:',ys)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss