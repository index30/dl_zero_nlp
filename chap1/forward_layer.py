'''forward処理のみを持つLayerの定義'''
import numpy as np
import chap1_activation

# Sigmoid関数の適用
class Sigmoid:
    def __init__(self):
        self.params = [] #学習するパラメータは存在しない

    def forward(self, x):
        return chap1_activation.sigmoid(x)

# 全結合層による変換
class Affine:
    def __init__(self, W, b):
        '''
        # Arguments
            W:  重み
            b:  バイアス
        '''
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out
