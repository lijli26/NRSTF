import IO
import time
import numpy as np
from sklearn.linear_model import LinearRegression

def segmentation(img, m):
    (n, h, w) = img.shape
    new_img = np.zeros((n, int(m * h), int(m * w)))
    for i in range(h):
        x = int(m * i)
        for j in range(w):
            y = int(m * j)

            new_img[:, x:x + m, y:y + m] = new_img[:, x:x + m, y:y + m] + img[:, i, j].reshape(n, 1, 1)

    return new_img

def aggerate(img, m):
    (n, h, w) = img.shape
    new_img = np.zeros((n, int(h // m), int(w // m)))

    for i in range(h // m):
        x = int(m * i)
        for j in range(w // m):
            y = int(m * j)
            cube = img[:, x:x + m, y:y + m]
            new_img[:, i, j] = np.sum(np.sum(cube, axis=1), axis=1) / m / m + new_img[:, i, j]

    return new_img

def reverse_padding(img, s):
    (h, w) = img.shape
    new_img = np.zeros((h + 2 * s, w + 2 * s))
    new_img[s:s + h, s:s + w] = img
    for i in range(s):
        new_img[i, s:s + h] = img[s - i, :]
        new_img[i + h + s, s:s + h] = img[h - s - i, :]

        new_img[s:s + w, i] = img[:, s - i]
        new_img[s:s + w, s + i + w] = img[:, w - s - i]

        new_img[i, 0:s] = img[s - i, 0:s]
        new_img[i, s + h:] = img[s - i, h - s:]
        new_img[i + h + s, 0:s] = img[h - s - i, 0:s]
        new_img[i + h + s, s + h:] = img[h - s - i, h - s:]

    return new_img

def denoise_M1(D_L1, D_M1):
    (n, h, w) = D_M1.shape
    D_L1 = np.concatenate((D_L1[0].reshape(1, h, w), D_L1[2:]), axis=0)
    D_M11 = np.concatenate((D_M1[0].reshape(1, h, w), D_M1[2:]), axis=0)

    m1_2b = np.zeros((2, h, w))

    ipt = np.zeros((h * w, 3))
    tar = np.zeros((h * w, 2))
    idx = 0
    A = np.zeros((2, 3))
    B = [0] * 2
    for i in range(h):
        for j in range(w):
            ipt[idx] = D_L1[0:3, i, j]
            tar[idx] = D_L1[3:, i, j]
            idx = idx + 1

    for t in range(2):
        sub_tar = tar[:, t]
        reg = LinearRegression()
        reg.fit(ipt, sub_tar)
        A[t, :] = reg.coef_
        B[t] = reg.intercept_

        m1_2b[t] = np.sum(D_M11[0:3] * reg.coef_.reshape(3, 1, 1), axis=0) + reg.intercept_

    P_m1 = np.concatenate((D_M1[0:4], m1_2b), axis=0)

    return P_m1

def kernal_cal(D_L1, D_M1, D_M2, s):
    D_L1_padding = reverse_padding(D_L1, s)
    D_M1_padding = reverse_padding(D_M1, s)

    (h, w) = D_L1_padding.shape
    (h1, w1) = D_L1.shape
    m1_2b = np.zeros(D_M1_padding.shape)

    ipt = np.zeros((h1 * w1, (2 * s + 1) ** 2))
    tar = np.zeros((h1 * w1, 1))
    idx = 0

    for i in range(s, h - s):
        for j in range(s, w - s):
            sub_DM1 = D_M1_padding[i - s:i + s + 1, j - s:j + s + 1]
            ipt[idx] = sub_DM1.reshape((2 * s + 1) ** 2, )

            tar[idx] = D_L1_padding[i, j]
            idx = idx + 1

    reg = LinearRegression()
    reg.fit(ipt, tar)

    i = j = 0
    D_M2_padding = reverse_padding(D_M2, s)
    for i in range(s, h - s):
        for j in range(s, w - s):

            sub_DM2 = D_M2_padding[i - s:i + s + 1, j - s:j + s + 1]

            m1_2b[i, j] = np.sum(sub_DM2.reshape((2 * s + 1) ** 2, ) * reg.coef_) + reg.intercept_


    m1_2b = m1_2b[s:s + h1, s:s + w1]
    m1_2b[m1_2b > 255] = 255
    m1_2b[m1_2b < 0] = 0
    return m1_2b

def Run():

    L1 = IO.imread(r'G:\L-2018-01-04.tif')
    L2 = IO.imread(r'G:\L-2018-05-12.tif')
    M1 = IO.imread(r'G:\M-2018-01-04.tif')
    M2 = IO.imread(r'G:\M-2018-05-12.tif')

    r = 20

    D_M1 = aggerate(M1, r)
    D_M2 = aggerate(M2, r)

    D_L1 = aggerate(L1, r)
    D_L2 = aggerate(L2, r)

    P_m1 = denoise_M1(D_L1 ,D_M1)
    P_m1[P_m1 > 255] = 255
    P_m1[P_m1 < 0] = 0

    IO.imsave(segmentation(P_m1, r), r'G:\P_m1.tif', 'float')

    for k in range(20):
        N_m1 = D_M1+0

        Noise1 = kernal_cal(abs(N_m1[4]-D_M1[4]), D_M1[4],D_M1[4],k+1)
        Noise2 = kernal_cal(abs(N_m1[5]-D_M1[5]), D_M1[5],D_M1[5],k+1)

        N_m1[4] = D_M1[4]-Noise1
        N_m1[5] = D_M1[5]-Noise2

        N_m1[4] = N_m1[4] / np.sum(N_m1[4]) * np.sum(D_M1[4])
        N_m1[5] = N_m1[5] / np.sum(N_m1[5]) * np.sum(D_M1[5])

        print(k+1)
        print(np.average(abs(N_m1[4]-D_L1[4])**2),np.average(abs(N_m1[5]-D_L1[5])**2))
    
    P_m2 = D_M2 + 0
    Noise1 = kernal_cal(abs(P_m1[4] -D_M1[4]), D_M1[4] ,D_M2[4] ,5)
    Noise2 = kernal_cal(abs(P_m1[5] -D_M1[5]), D_M1[5] ,D_M2[5] ,4)

    P_m2[4] = D_M2[4] -Noise1
    P_m2[5] = D_M2[5] -Noise2

    IO.imsave(segmentation(P_m2, r), r'G:\P_m2.tif', 'float')

start = time.clock()
Run()
end = time.clock()
print('Running time:%s Seconds' % (end - start))
