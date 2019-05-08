# coding: utf-8
import numpy as np
from function import softmax
from rnnlm import Rnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=500):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            '''if fl.pas(12):
                print('score.in.x:{}'.format(x))
                print('score.in.x:{}'.format(x.shape))
            #(1,1)=[[316]]'''
            score = self.predict(x)
            #if fl.pas(12):
             #   print('score:{}'.format(score.shape))
              #  print('score:{}'.format(score))
            p = softmax(score.flatten())
            #if fl.pas(13):
             #   print('score.flatten:{}'.format(score.flatten().shape))
              #  print('score.flatten:{}'.format(score.flatten()))
               # print('p:{}'.format(p.shape))
                #print('p:{}'.format(p))
            sampled = np.random.choice(len(p), size=1, p=p)
            #print('sampled:{}'.format(sampled))

            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids