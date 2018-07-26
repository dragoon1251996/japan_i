import json
import math

import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt
import mysql.connector as connect
import numpy as np

cnt = connect.connect(user='root', password='', host='127.0.0.1', database='net3')
cursor = cnt.cursor()


def getdata():
    cursor.execute("SELECT sample,input from kanji")
    temp = cursor.fetchall()
    temp=[(i[0], json.loads(i[1])) for i in temp]
    # k = []
    # for i in temp:
    #     t = [
    #         (i[0], json.loads(i[1])),
    #         (i[0], list(json.loads(i[1]))[::-1])
    #     ]
    #     if [i[0], np.random.permutation(list(json.loads(i[1]))).tolist()] not in t:
    #         t.append((i[0], np.random.permutation(list(json.loads(i[1]))).tolist()))
    #
    #     for j in t:
    #         k.append(j)
    # return k
    return temp

def figdata(data):
    k = [X[1] for X in data]
    label = [X[0] for X in data]
    return [k, label]


# for i in figdata(getdata())[0]:
#     print(i)


def remove1(S, T, L):
    R = []
    for i in S:
        temp = []
        temp.append(i[0])
        for j in range(1, len(i) - 1):
            x = i[j][0] - temp[-1][0]
            y = i[j][1] - temp[-1][1]

            if T * T < x * x + y * y:
                # if (Dx*x+Dy*y)/(math.sqrt(x*x+y*y)*math.sqrt(Dx*Dx+Dy*Dy))>Tc:
                temp.append(i[j])
        temp.append(i[-1])
        R.append(temp)
    return [R, L]


def size(S):
    maxX = -10000
    minX = 10000
    maxY = -10000
    minY = 10000
    for i in S:
        for j in i:
            if j[0] > maxX:
                maxX = j[0]
            if j[0] < minX:
                minX = j[0]
            if j[1] > maxY:
                maxY = j[1]
            if j[1] < minY:
                minY = j[1]
    return max([maxX - minX, maxY - minY])


def remove2(S, Td, T, L):
    S = remove1(S, Td, L)[0]
    R = []
    for i in S:
        temp = []
        temp.append(i[0])
        for j in range(1, len(i) - 1):
            x = i[j][0] - temp[-1][0]
            y = i[j][1] - temp[-1][1]
            Dx = i[j + 1][0] - i[j][0]
            Dy = i[j + 1][1] - i[j][1]
            if (Dx * x + Dy * y) / (math.sqrt(x * x + y * y) * math.sqrt(Dx * Dx + Dy * Dy) + 0.1) < T:
                temp.append(i[j])
        temp.append(i[-1])
        R.append(temp)
    return [R, L]


def infor(data, index):
    try:
        tem = []
        # MINX=-100000
        # MINY=-100000
        # for i in data[0][index]:
        #     le = len(i[0])
        #     for j in range(le):
        #         if MINX>i[0][j]:
        #             MINX=i[0][j]
        #         if MINY>i[1][j]:
        #             MINY=i[1][j]

        for i in data[0][index]:
            le = len(i[0])
            l = []
            for j in range(le):
                l.append([i[0][j], i[1][j]])
            tem.append(l)
        return [tem, data[1][index]]
    except:
        return 0


def show(kanji):
    for i in kanji[0]:
        for j in i:
            plt.plot(j[0], -j[1], "ro")
    for i in kanji[0]:
        list_of_lists = i
        x_list = [x for [x, y] in list_of_lists]
        y_list = [-y for [x, y] in list_of_lists]
        plt.plot(x_list, y_list)
    font_path = "SimHei.ttf"
    prop = mfm.FontProperties(fname=font_path, size=20)
    plt.title(kanji[1], fontproperties=prop)
    # plt.xlim(plt.xlim(-4.0, 4.0))
    # plt.ylim(plt.xlim(-4.0, 4.0))
    plt.show()


def Len(x1, y1, x2, y2):
    return math.sqrt(math.pow(abs(x1 - x2), 2) + math.pow(abs(y1 - y2), 2))


def P(x1, y1, x2, y2):
    A = 1 / 2 * Len(x1, y1, x2, y2)
    return [A * (x1 + x2), A * (y1 + y2), 2 * A]


# def Micro(L):
#     sum_L=0
#     sum_Px=0
#     sum_Py=0
#     for i in range(len(L)-1):
#         p=P(L[i][0],L[i][1],L[i+1][0],L[i+1][1])
#         sum_L=sum_L+p[2]
#         sum_Px=sum_Px+p[0]
#         sum_Py=sum_Py+p[1]
#     return [sum_Px/sum_L,sum_Py/sum_L]
# def dXL(x1,y1,x2,y2,microX):
#     return 1/3*Len(x1,y1,x2,y2)*(math.pow(abs(x2-microX),2)+math.pow(abs(x1-microX),2)+(x1-microX)*(x2-microX))
# def deltaX(L,microX):
#     sum_Dxl=0
#     sum_len=0
#     for i in range(len(L)-1):
#         sum_Dxl=sum_Dxl+dXL(L[i][0],L[i][1],L[i+1][0],L[i+1][1],microX)
#         sum_len=sum_len+Len(L[i][0],L[i][1],L[i+1][0],L[i+1][1])
#     return math.sqrt(sum_Dxl/sum_len)


def Micro(L):
    sum_L = 0
    sum_Px = 0
    sum_Py = 0
    for j in range(len(L)):
        for i in range(len(L[j]) - 1):
            p = P(L[j][i][0], L[j][i][1], L[j][i + 1][0], L[j][i + 1][1])
            sum_L = sum_L + p[2]
            sum_Px = sum_Px + p[0]
            sum_Py = sum_Py + p[1]
    return [sum_Px / sum_L, sum_Py / sum_L]


def dXL(x1, y1, x2, y2, microX):
    return 1 / 3 * Len(x1, y1, x2, y2) * (
        math.pow(abs(x2 - microX), 2) + math.pow(abs(x1 - microX), 2) + (x1 - microX) * (x2 - microX))


def deltaX(L, microX):
    sum_Dxl = 0
    sum_len = 0
    for j in range(len(L)):
        for i in range(len(L[j]) - 1):
            sum_Dxl = sum_Dxl + dXL(L[j][i][0], L[j][i][1], L[j][i + 1][0], L[j][i + 1][1], microX)
            sum_len = sum_len + Len(L[j][i][0], L[j][i][1], L[j][i + 1][0], L[j][i + 1][1])
    return math.sqrt(sum_Dxl / sum_len)


def resizeArr(A):
    L = []
    for i in A:
        for j in i:
            L.append(j)
    return L


def nomal(A, delta, micro):
    K = []
    t = len(A[0])
    for i in range(t):
        temp = []
        for j in range(len(A[0][i])):
            # print("in",A[0][i][j][0],"   ",A[0][i][j][1]," ",i," ",j)
            # A[0][i][j][0]=(A[0][i][j][0]-micro[0])/delta
            # A[0][i][j][1]=(A[0][i][j][1]-micro[1])/delta
            # if j!=len(A[0][i])-1:
            #     print(A[0][i][j + 1][0], "   ", A[0][i][j + 1][1])

            temp.append([(A[0][i][j][0] - micro[0]) / delta, (A[0][i][j][1] - micro[1]) / delta])
        K.append(temp)
    return [K, A[1]]
    # return A


def feature(net, label):
    F = []
    l1 = len(net)
    for i in range(l1):
        l2 = len(net[i])
        for j in range(l2):
            if j != l2 - 1:
                F.append(
                    [net[i][j][0], net[i][j][1], net[i][j + 1][0] - net[i][j][0], net[i][j + 1][1] - net[i][j][1], 1,
                     0])
            else:
                if i == l1 - 1:
                    continue
                else:
                    F.append(
                        [net[i][j][0], net[i][j][1], net[i + 1][0][0] - net[i][j][0], net[i + 1][0][1] - net[i][j][1],
                         0, 1])

    return [F, label]


from collections import Counter


def dict_data():
    data = figdata(getdata())
    T = dict(Counter(data[1]))
    d = {}
    j = 1
    for i in T:
        if T[i] > 50:
            d[i] = j
            j += 1
    with open('dict.json', 'w', encoding="utf-8") as outfile:
        json.dump(d, outfile, indent=4, ensure_ascii=False)


# dict_data()
def read_dict():
    with open("dict.json", encoding="utf-8") as data_file:
        data = json.load(data_file)
    return data


def gen_L(i, length):
    tem = []
    for j in range(1, length + 1):
        if j == i:
            tem.append(1)
        else:
            tem.append(0)
    return tem


# dict_data()(*****)
def dataset():
    data = figdata(getdata())
    # print(len(data[0]))
    # print(Counter(data[1]))
    # with open('data.txt', 'w',encoding="utf-8") as outfile:
    #     json.dump(dict(Counter(data[1])), outfile,indent=4,ensure_ascii=False)
    # print(len(list(set(data[1]))))
    erro = 0
    temp = []

    for i in range(len(data[0])):
        try:
            kanji = infor(data, i)
            siz = size(kanji[0])
            kanji = remove2(kanji[0], 0.075 * siz, 0.99, kanji[1])
            micro = Micro(kanji[0])
            delta = deltaX(kanji[0], micro[0])
            K = nomal(kanji, delta, micro)
            # print(feature(K[0],K[1]))



            # temp.append(feature(K[0],K[1]))



            # if K[1]=="手":
            #     show(K)
            dict = read_dict()
            length = len(dict)
            for i in dict:
                if K[1] == i:
                    temp.append([feature(K[0], K[1])[0], gen_L(dict[i], length)])


                    # if K[1] == "入":
                    #     temp.append([feature(K[0], K[1])[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                    # if K[1] == "る":
                    #     temp.append([feature(K[0], K[1])[0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                    # if K[1] == "す":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                    # if K[1] == "付":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
                    # if K[1] == "出":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
                    # if K[1] == "し":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
                    # if K[1] == "業":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
                    # if K[1] == "通":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
                    # if K[1] == "上":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
                    # if K[1] == "前":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
                    # if K[1] == "い":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
                    # if K[1] == "定":
                    #     temp.append([feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        except:
            erro += 1
            continue
    # print(erro,"sdsd")
    return temp


def test_data(X, l):
    data = ""
    kanji = []
    t = []
    for i in X:
        t.append([[i[0][j], i[1][j]] for j in range(len(i[0]))])
    kanji = [t, l]
    siz = size(kanji[0])
    kanji = remove2(kanji[0], 0.075 * siz, 0.99, kanji[1])
    micro = Micro(kanji[0])
    delta = deltaX(kanji[0], micro[0])
    print(delta)
    K = nomal(kanji, delta, micro)
    dict = read_dict()
    length = len(dict)

    for i in dict:
        if K[1] == i:
            data = [feature(K[0], K[1])[0], gen_L(dict[i], length)]
    # if K[1] == "入":
    #     data=[feature(K[0], K[1])[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # if K[1] == "る":
    #     data =[feature(K[0], K[1])[0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # if K[1] == "す":
    #     data =[feature(K[0], K[1])[0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # if K[1] == "付":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    # if K[1] == "出":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
    # if K[1] == "し":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
    # if K[1] == "業":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    # if K[1] == "通":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    # if K[1] == "上":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
    # if K[1] == "前":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]
    # if K[1] == "い":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
    # if K[1] == "定":
    #     data =[feature(K[0], K[1])[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    return data

# print(test_data([[[214,212,210,207,205,201,196,193,189,187,182,180,178,176,174,173,171,167,165,161,158,153,150,147,144,139,136,134,131,130,129],[163,163,163,165,167,169,171,174,176,178,180,182,183,184,185,186,187,188,188,188,189,190,191,192,192,194,195,196,197,197,197],[0,104,112,120,128,136,144,152,160,168,176,183,192,199,208,217,224,232,240,248,256,264,272,280,288,297,304,312,320,328,336]],[[198,199,199,199,200,200,201,201,202,202,203,204,205,206,207,208,209,212,214,217,220,226,229,233,236,238,239,241,242,243,244,246,249,251,252,254,255,258,259],[96,96,98,100,105,109,114,119,123,126,133,138,144,149,154,159,164,173,177,183,189,198,204,209,215,217,219,220,223,223,225,227,229,231,233,234,238,241,244],[1088,1112,1120,1128,1136,1144,1152,1160,1168,1176,1184,1192,1200,1207,1217,1224,1232,1240,1248,1256,1263,1271,1280,1287,1296,1303,1311,1319,1327,1335,1343,1351,1359,1367,1376,1383,1391,1399,1407]]],"入"))
# show([[[[0.3539069468916596, -0.44617460247854784], [-1.168174898637568, 0.36344340046253065], [-2.3987942631080075, 0.6549058815213189]], [[-0.16424857499063064, -2.615950850360638], [0.1919833463034439, -0.4137898823609047], [1.0663707894798087, 1.2378308436388954], [1.5845263113620989, 1.820755805756472], [1.8112193521856008, 2.1769877270505464]]], '入'])
# print(dataset()[0])
# print(test_data([[[109,111,116,121,131,141,155,170,187,203,217,233,243,251,256,259,260],[73,73,73,72,71,69,66,63,58,55,53,50,46,44,42,41,39],[0,32,40,48,56,64,72,80,88,96,104,112,120,128,136,145,152]],[[187,188,189,189,191,192,193,195,196,198,198,201,203,203,205,206,206,207,208,208,210,211,211,211,212,212,213,213,213,213,213,213,213,213,213,213,213,213,213,213,213,213,213,212,212,211,210,208,208,206,205,204,203,202,202,201,198,198,197,196,195,193,191,189,186,183,181,177,176,174,173,172],[64,64,64,66,67,71,73,75,77,81,84,87,90,92,95,98,100,102,105,106,109,111,112,114,118,120,122,123,124,127,128,131,132,133,134,137,138,139,141,142,144,147,148,150,151,156,157,158,161,163,167,169,171,175,176,177,179,181,181,182,184,185,186,186,186,186,186,186,186,186,186,185],[1496,1584,1600,1617,1624,1632,1640,1648,1656,1665,1672,1680,1688,1696,1704,1712,1720,1728,1736,1745,1752,1761,1768,1776,1784,1793,1800,1808,1816,1825,1832,1840,1848,1856,1864,1872,1880,1888,1896,1904,1920,1936,1945,1962,1968,1976,1984,1992,2000,2008,2016,2024,2032,2040,2048,2056,2064,2072,2080,2088,2096,2104,2112,2120,2128,2136,2145,2152,2162,2176,2193,2224]],[[166,167,172,179,192,206,222,241,262,279,291,299,302,303],[95,95,95,95,95,95,95,95,95,92,90,89,87,87],[2665,2704,2712,2720,2728,2736,2745,2752,2762,2768,2776,2784,2792,2808]],[[197,201,205,210,217,227,239,252,265,275,284,292,294,296],[143,143,142,141,140,139,139,139,139,139,139,138,137,137],[3200,3232,3240,3248,3256,3264,3272,3280,3288,3296,3304,3312,3320,3328]]],"手"))
# show([[[[-2.525296557701641, -0.7189998385687064], [0.9074605014217211, -1.4200558576854494], [1.1250296108027793, -1.5409275851193707]], [[-0.6396976097324703, -0.9365689479497646], [-0.25290808194392245, -0.30803596529337435], [-0.03533897256286426, 0.41719439931015284], [-0.011164627076080015, 0.8281582725854849], [-0.059513318049648496, 1.2874708368343855], [-0.4221285003514121, 1.9160038194907758], [-0.7847436826531757, 2.012701201437913], [-1.0023127920342338, 1.9885268559511284]], [[-1.1473588649549393, -0.18716423785945316], [1.1733783017763477, -0.18716423785945316], [2.067829084787365, -0.3322103107801586], [2.1645264667345017, -0.3805590017537271]], [[-0.3979541548646279, 0.9732043455061904], [1.995306048327012, 0.8281582725854849]]], '手'])
# for i in range(1000,2000):
#     try:
#         t=time.time()
#         kanji=infor(data,i)
#         siz = size(kanji[0])
#         kanji=remove2(kanji[0],0.05*siz,0.99,kanji[1])
#         kanji_1D=resizeArr([X for X in kanji[0]])
#         micro=Micro(kanji_1D)
#         delta=deltaX(kanji_1D,micro[0])
#         K=nomal(kanji,delta,micro)
#         print(time.time()-t)
#         print(K)
#         show(K)
#     except:
#         continue
