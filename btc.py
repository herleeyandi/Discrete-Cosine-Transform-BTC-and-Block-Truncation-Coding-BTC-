"""
Block Truncation Coding (BTC) Algorithm
Herleeyandi Markoni
12/04/2017
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def btc_block(img_block):
    """
    This function will calculate the coefficient and bitmap of the BTC Algorithm.
    """
    m = img_block.shape[0] * img_block.shape[1]
    h = float(np.sum(img_block) / (img_block.shape[0] * img_block.shape[1]))
    h2 = float(np.sum(img_block ** 2) / (img_block.shape[0] * img_block.shape[1]))
    sigma2 = np.var(img_block)
    sigma = np.std(img_block)
    q = img_block[np.where(img_block > h)].shape[0]
    a = h - sigma * math.sqrt(float(q) / (m - q))
    b = h + sigma * math.sqrt((m - q) / float(q))
    bitmap = np.zeros((img_block.shape[0], img_block.shape[1]))
    result = img_block.copy()
    for i in range(0, img_block.shape[0]):
        for j in range(0, img_block.shape[1]):
            if (img_block[i, j] >= h):
                bitmap[i, j] = 1
                result[i, j] = round(b, 0)
            else:
                result[i, j] = round(a, 0)
    return bitmap, result


def btc_manual(img, block):
    """
    This function will doing Block Truncation Coding of one entire image.
    """
    result = img.copy()
    bitmap = img.copy()
    count = 0
    for i in range(0, img.shape[0], block):
        for j in range(0, img.shape[1], block):
            bitmap[i: i + block, j: j + block], result[i: i + block, j: j + block] = btc_block(
                img[i: i + block, j: j + block])
            count += 1
    return bitmap, result

def btc_psnr(img1, img2):
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