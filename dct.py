"""
Discrete Cosine Transform (DCT) Algorithm
Herleeyandi Markoni
12/04/2017
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

PI = 3.141592


def get_c(value):
    """
    This function will get the coefficient of Ci and Cj.
    """
    if (value == 0):
        return float(1.0 / math.sqrt(2.0))
    else:
        return float(1.0)


def dct_block(img_block):
    """
    This function will transform one block into frequency domain using original DCT Algorithm.
    """
    result = np.zeros((img_block.shape[0], img_block.shape[1]), dtype=np.float64)
    N = img_block.shape[0]
    for i in range(0, img_block.shape[0]):
        for j in range(0, img_block.shape[1]):
            ci = get_c(i)
            cj = get_c(j)
            tmp = 0
            for k in range(0, img_block.shape[0]):
                for l in range(0, img_block.shape[1]):
                    cal = float(img_block[k, l] * math.cos(
                        float((2.0 * float(k) + 1.0) * float(i * math.pi)) / float(2.0 * N)) * math.cos(
                        float((2.0 * float(l) + 1.0) * float(j * math.pi)) / float(2.0 * N)))
                    tmp += cal
            result[i, j] = float(2.0 / N) * ci * cj * tmp
    return result


def dct_manual(img, block):
    """
    This function will transform entire image image into frequency domain using original DCT Algorithm.
    """
    result = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0], block):
        for j in range(0, img.shape[1], block):
            result[i: i + block, j: j + block] = dct_block(img[i: i + block, j: j + block])
    return result


def idct_block(img_block):
    """
    This function will convert back one block of image from frequency domain to spatial domain using inverse-DCT Transform Algorithm.
    """
    result = np.zeros((img_block.shape[0], img_block.shape[1]))
    N = img_block.shape[0]
    for i in range(0, img_block.shape[0]):
        for j in range(0, img_block.shape[1]):

            tmp = 0
            for k in range(0, img_block.shape[0]):
                for l in range(0, img_block.shape[1]):
                    ci = get_c(k)
                    cj = get_c(l)
                    cal = float(ci * cj * img_block[k, l] * math.cos(
                        float((2.0 * float(i) + 1.0) * float(k * math.pi)) / float(2.0 * N)) * math.cos(
                        float((2.0 * float(j) + 1.0) * float(l * math.pi)) / float(2.0 * N)))
                    tmp += cal
            result[i, j] = round(float(2.0 / N) * tmp)
    return result


def idct_manual(img, block):
    """
    This function will convert entire image from frequency domain to spatial domain using inverse-DCT Transform Algorithm.
    """
    result = np.zeros((img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0], block):
        for j in range(0, img.shape[1], block):
            result[i: i + block, j: j + block] = idct_block(img[i: i + block, j: j + block])
    return result

def dct_1d(inp, block):
    """
    This function will doing 1-Dimensional DCT Transform.
    """
    out = np.zeros((inp.shape[0],), dtype=np.float32)
    for u in range(0,block):
        z = 0
        for x in range(0,block):
            z += inp[x] * math.cos(PI * float(u) * float((2.0 * float(x) + 1))/float(2.0 * block))
        out[u] = float(z * get_c(u)/2.0)
    return out

def fast_dct_block(img_block):
    """
    This function will convert one block image into frequency domain using Fast-DCT Algorithm.
    """
    temp = np.zeros((img_block.shape[0], img_block.shape[1]), dtype = np.float64)
    result = np.zeros((img_block.shape[0], img_block.shape[1]), dtype = np.float64)
    N = img_block.shape[0]
    for i in range(0,img_block.shape[0]):
        temp[i,:] = dct_1d(img_block[i,:], N)
    for j in range(0, img_block.shape[1]):
        result[:,j] = dct_1d(temp[:,j], N)
    return result

def fast_dct_manual(img, block):
    """
    This function is Fast-DCT Algorithm which separate rows and column operation.
    """
    result = np.zeros((img.shape[0], img.shape[1]))
    count = 0
    for i in range(0, img.shape[0], block):
        for j in range(0, img.shape[1], block):
            result[i : i + block, j: j + block] = fast_dct_block(img[i : i + block, j: j + block])
            count+=1
    return result

def dct_psnr(img1, img2):
    """
    This function will calculate the PSNR of two images.
    """
    mse = np.mean( (img1 - img2) ** 2 )
    psnr = 0
    if mse == 0:
        psnr = 100
    else:
        PIXEL_MAX = 255.0
        20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    print('PSNR: {}'.format(psnr))
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Original")
    ax1.imshow(img1, cmap='gray')

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title("Result")
    ax2.imshow(img2, cmap='gray')

    fig.set_figheight(7)
    fig.set_figwidth(14)
    plt.show()

def plot_save(image, name, size):
    """
    This function will plot and save the image.
    """
    cv2.imwrite(name, image)
    fig = plt.gcf()
    fig.set_figheight(size)
    fig.set_figwidth(size)
    plt.imshow(image, cmap='gray')
    plt.show()