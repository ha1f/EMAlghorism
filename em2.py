#coding: utf-8

#はるふ
#plot以外はベクトルの次元が違ってもいけるはず

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random


M = 10 #混合数
sigma2 = 0.1#正規分布の分散

#ベクトルの長さの2乗(スカラー)
vecLen2 = lambda vec: sum([cell ** 2 for cell in vec])
#ベクトルの差(スカラー)
vecDiff = lambda vec1, vec2: [a-b for a,b in zip(vec1, vec2)]
#ベクトルの和
vecAdd = lambda vec1, vec2: [a+b for a,b in zip(vec1, vec2)]
#ベクトルの実数倍
vecMulti = lambda vec1, alp: [alp*i for i in vec1]
#距離の二乗
dist2 = lambda vec1, vec2: vecLen2(vecDiff(vec1, vec2))

#dataに入った点をすべてプロット
def plotData(ax, data, option):
    xData, yData, zData = list(zip(*data))#行列の縦横入れ替え
    ax.plot(xData, yData, zData, option)
    return

def show(ax):
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()
    return

#[[x1,y1,z1], [x2,y2,z2], ……]となっている点のデータを作成
def getData(fileName):
    data = []
    f = open(fileName, 'r')
    data = [[float(i) for i in line.split()] for line in f]
    f.close()
    return data

#Estep
#中心位置パラメータを仮定して、事後分布を計算する
def Estep(data, mu):
    global M
    gaus = lambda i,x,mu: math.exp(-1.0/2.0/sigma2*dist2(x,mu[i]))
    sum1i=[0.0]*M
    vecxi=[[0,0,0]]*M
    for x in data:#xはN次元vector、mu[j]も
        currentSum = sum([gaus(j,x,mu) for j in range(0, M)])
        Pxi = [gaus(i,x,mu)/currentSum for i in range(0, M)]
        for i in range(0, M):
            sum1i[i] = sum1i[i] + Pxi[i]
            vecxi[i] = vecAdd(vecxi[i], vecMulti(x, Pxi[i]))
    return (sum1i, vecxi)

#Mstep
#中心位置パラメータの更新
def Mstep(sum1i, vecxi):
    return [vecMulti(x, 1.0/one) for one,x in zip(sum1i, vecxi)]

#dataのうちからランダムに重複なくnum個選んで返す
def choiceRandom(data, num):
    return random.sample(data, num)

def main():
    #読み込み、今回はT=200, N=3
    data = getData('em.data')
    #初期の中心位置を選択
    mu = choiceRandom(data, M)
    #学習を進行させる
    for i in range(0,100):
        sum1i, vecxi = Estep(data, mu)
        mu = Mstep(sum1i, vecxi)
        print(i)
    #描画
    print(mu)
    ax = Axes3D(plt.figure())
    plotData(ax, mu, 'bo')
    plotData(ax, data, 'r.')
    show(ax)

main()
