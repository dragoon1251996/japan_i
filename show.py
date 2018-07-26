import matplotlib.pyplot as plt
import re
data=open("net.txt",encoding="utf-8").read().split("WORD")
import math
def infor(d,index):
    label=d[index].strip()[0]
    stroke=re.findall("POINTS(.*?)\n",d[index])
    S=[]
    for i in stroke:
        temp=i.split("#")[1].strip().split(" ")
        x=[]
        y=[]
        for j in range(len(temp)):
           if j%2==1:
               x.append(2000-int(temp[j]))
           else:
               y.append(int(temp[j]))
        S.append([[y[t],x[t]] for t in range(len(x))])
    return [S,label]

def size(S):
    maxX=-10000
    minX=10000
    maxY=-10000
    minY=10000
    for i in S:
        for j in i:
            if j[0]>maxX:
                maxX=j[0]
            if j[0]<minX:
                minX=j[0]
            if j[1]>maxY:
                maxY=j[1]
            if j[1]<minY:
                minY=j[1]
    return max([maxX-minX,maxY-minY])



def remove1(S,T):
    R=[]
    for i in S:
        temp = []
        temp.append(i[0])
        for j in range(1,len(i)):
            x=i[j][0]-temp[-1][0]
            y=i[j][1]-temp[-1][1]


            if T*T<x*x+y*y :
                # if (Dx*x+Dy*y)/(math.sqrt(x*x+y*y)*math.sqrt(Dx*Dx+Dy*Dy))>Tc:
                temp.append(i[j])
        # temp.append(i[-1])
        R.append(temp)
    return R

def remove2(S,Td,T):
    S=remove1(S,Td)
    R = []
    for i in S:
        temp = []
        temp.append(i[0])
        for j in range(1, len(i) - 1):
            x = i[j][0] - temp[-1][0]
            y = i[j][1] - temp[-1][1]
            Dx = i[j + 1][0] - i[j][0]
            Dy = i[j + 1][1] - i[j][1]
            if (Dx*x+Dy*y)/(math.sqrt(x*x+y*y)*math.sqrt(Dx*Dx+Dy*Dy))<T:
                temp.append(i[j])
        temp.append(i[-1])
        R.append(temp)
    return R


import numpy as np
def resize(S):
    return [X for X in S[i]]

import time
t=time.time()
k=infor(data,10000)
size=size(k[0])
print(k[1])
print(k[0])
print(remove2(k[0],0.05*size,0.99))
# plt.plot(k[0][0],k[0][1],"ro")

for i in remove2(k[0],0.05*size,0.99):
    for j in i:
        plt.plot(j[0],j[1],"ro")

for i in remove2(k[0],0.05*size,0.99):
    list_of_lists = i
    x_list = [x for [x, y] in list_of_lists]
    y_list = [y for [x, y] in list_of_lists]
    plt.plot(x_list, y_list)



plt.axis([-1000, 2000, -1000, 2000])




def Len(x1,y1,x2,y2):
    return math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
def P(x1,y1,x2,y2):
    A=1/2*Len(x1,y1,x2,y2)
    return [A*(x1+x2),A*(y1+y2),2*A]
def Micro(L):
    sum_L=0
    sum_Px=0
    sum_Py=0
    for i in range(len(L)-1):
        p=P(L[i][0],L[i][1],L[i+1][0],L[i+1][1])
        sum_L=sum_L+p[2]
        sum_Px=sum_Px+p[0]
        sum_Py=sum_Py+p[1]
    return [sum_Px/sum_L,sum_Py/sum_L]
def dXL(x1,y1,x2,y2,microX):
    return 1/3*Len(x1,y1,x2,y2)*(math.pow(x2-microX,2)+math.pow(x1-microX,2)+(x1-microX)*(x2-microX))
def deltaX(L,microX):
    sum_Dxl=0
    sum_len=0
    for i in range(len(L)-1):
        sum_Dxl=sum_Dxl+dXL(L[i][0],L[i][1],L[i+1][0],L[i+1][1],microX)
        sum_len=sum_len+Len(L[i][0],L[i][1],L[i+1][0],L[i+1][1])
    return math.sqrt(sum_Dxl/sum_len)




microX=Micro(remove2(k[0],0.05*size,0.99)[0])
delta=deltaX(remove2(k[0],0.05*size,0.99)[0],microX[0])
i=np.array(remove2(k[0],0.05*size,0.99))
i=i/delta
print(i)

plt.show()