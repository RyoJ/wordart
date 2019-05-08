# coding: utf-8
import numpy as np
from time_layers import *
#from base_model import BaseModel


class Rnnlm:#(BaseModel):
    def __init__(self, vocab_size=700, wordvec_size=100, hidden_size=100):#vocab_sizeは辞書の語彙数より小さくする,小さすぎるとIndexError: index 158 is out of bounds for axis 0 with size 150
        V, D, H = vocab_size, wordvec_size, hidden_size
        #print('rnnlmV:{}'.format(V))
        #print('rnnlmD:{}'.format(D))
        #print('rnnlmH:{}'.format(H))
        rn = np.random.randn

        # 重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        print('embed_W:',embed_W.shape)
        print('lstm_Wx:',lstm_Wx.shape)
        print('lstm_Wh:',lstm_Wh.shape)
        print('lstm_b:',lstm_b.shape)
        print('affine_W:',affine_W.shape)
        print('affine_b:',affine_b.shape)
        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        #if ch07.flag.get_flag(8)==False:
         #   print('layers:{}'.format(self.layers[:]))

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            hs = layer.forward(xs)
            #if fl.pas(14):
             #   print('precict.in.xs:{}'.format(xs.shape))
              #  print('precict.in.xs:{}'.format(xs))
               # print('precict.out.hs:{}'.format(hs.shape))
                #print('precict.hs:{}'.format(hs))
            xs=hs
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss