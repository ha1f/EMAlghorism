#coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import random

"""
潜在変数を求めたい（本来何であったか）


"""



M = 5 #混合数
N = 3 #ベクトルの次元
sigma2 = 0.1#分散
T = 200

fig = plt.figure()
ax = Axes3D(fig)

def plotData(data, option):
    xData = [datum[0] for datum in data]
    yData = [datum[1] for datum in data]
    zData = [datum[2] for datum in data]
    ax.plot(xData, yData, zData, option)
    return

def show():
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    plt.show()

def getData(fileName):
    data = []
    f = open(fileName, 'r')
    for line in f:
        data.append([float(i) for i in line.split()])
    f.close()
    return data

#正規分布
gaus = lambda i,x,mu: math.exp(-1.0/2.0/sigma2*dist2(x,mu[i]))
#ベクトルの長さの2乗(スカラー)
vecLen2 = lambda vec: sum([cell ** 2 for cell in vec])
#ベクトルの差(スカラー)
vecDiff = lambda vec1, vec2: [a-b for a,b in zip(vec1, vec2)]
#距離の二乗
dist2 = lambda vec1, vec2: vecLen2(vecDiff(vec1, vec2))
#ベクトルの和
vecAdd = lambda vec1, vec2: [a+b for a,b in zip(vec1, vec2)]
#ベクトルの実数倍
vecMulti = lambda vec1, alp: [alp*i for i in vec1]

#中心位置パラメータを仮定して、事後分布を計算する
def Estep(data, mu):
    global gaus
    global M
    global T
    sum1i=[0.0]*M
    vecxi=[[0,0,0]]*M
    for x in data:#xはN次元vector、mu[j]も
        currentSum = sum([gaus(j,x,mu) for j in range(0, M)])
        Pxi = [gaus(i,x,mu)/currentSum for i in range(0, M)]
        for i in range(0, M):
            sum1i[i] = sum1i[i] + Pxi[i]
            vecxi[i] = vecAdd(vecxi[i], vecMulti(x, Pxi[i]))
    return (sum1i, vecxi)

#中心位置パラメータの更新
def Mstep(sum1i, vecxi):
    return [vecMulti(x, 1.0/one) for one,x in zip(sum1i, vecxi)]

#dataのうちからランダムにnum個選んで返す
def choiceRandom(data, num):
    return random.sample(data, num)


def main():
    #読み込み
    data = getData('em.data')
    #初期の中心位置を選択
    mu = choiceRandom(data, M)
    #学習を進行させる
    for i in range(0,100):
        sum1i, vecxi = Estep(data, mu)
        mu = Mstep(sum1i, vecxi)
        print(mu)
    plotData(mu, 'bo')
    plotData(data, 'r.')
    show()
main()
