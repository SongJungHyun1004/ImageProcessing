import plistlib

import numpy as np
import cv2
import time

def Quantization_Luminance(scale_factor):
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance * scale_factor

def img2block(src, n=8):
    ######################################
    # TODO                               #
    # img2block 완성                      #
    # img를 block으로 변환하기              #
    # blocks : 각 block을 추가한 list       #
    # return value : np.array(blocks)     #
    ######################################
    blocks = list()
    (h, w) = src.shape
    for i in range(0, h, n):
        for j in range(0, w, n):
            blocks.append(src[i:i+n, j:j+n])
    return np.array(blocks)


def DCT(block, n=8):
    ######################################
    # TODO                               #
    # DCT 완성                            #
    ######################################
    def C(w, n):
        if w == 0:
            return (1 / n) ** 0.5
        else:
            return (2 / n) ** 0.5
    dst = np.zeros(block.shape)
    v, u = dst.shape
    y, x = np.mgrid[0:u, 0:v]
    for v_ in range(v):
        for u_ in range(u):
            mask = np.cos((2*x+1)*u_*np.pi/(2*n))*np.cos((2*y+1)*v_*np.pi/(2*n))
            dst[v_, u_] = C(v_, n)*C(u_, n)*np.sum(block[y, x]*mask)
    return np.round(dst)

def my_zigzag_encoding(block,block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_encoding 완성             #
    ######################################
    zigzag = list()
    zero = list()
    i, j = 0, 0
    while i < block_size and j < block_size:
        data = block[i, j]
        if (i+j) % 2 == 0:
            if i == 0:
                if j + 1 == block_size:
                    i = i + 1
                else:
                    j = j + 1
            elif j + 1 == block_size and i < block_size:
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        else:
            if j == 0:
                if i + 1 == block_size:
                    j = j + 1
                else:
                    i = i + 1
            elif i + 1 == block_size and j < block_size:
                j = j + 1
            else:
                i = i + 1
                j = j - 1
        if data == 0:
            zero.append(data)
        else:
            zigzag.extend(zero)
            zigzag.append(data)
            zero.clear()
        if i == block_size-1 and j == block_size-1:
            data = block[i, j]
            if data == 0:
                zero.append(data)
            else:
                zigzag.extend(zero)
                zigzag.append(data)
                zero.clear()
            break
    if len(zero) > 0:
        zigzag.append('EOB')

    return zigzag

def my_zigzag_decoding(block, block_size=8):
    ######################################
    # TODO                               #
    # my_zigzag_decoding 완성             #
    ######################################
    re_block = np.zeros((block_size, block_size))
    i, j = 0, 0
    n = 0
    while n < len(block):
        if block[n] == 'EOB':
            break
        re_block[i, j] = block[n]
        n = n + 1
        if (i + j) % 2 == 0:
            if i == 0:
                if j + 1 == block_size:
                    i = i + 1
                else:
                    j = j + 1
            elif j + 1 == block_size and i < block_size:
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        else:
            if j == 0:
                if i + 1 == block_size:
                    j = j + 1
                else:
                    i = i + 1
            elif i + 1 == block_size and j < block_size:
                j = j + 1
            else:
                i = i + 1
                j = j - 1
    return re_block

def DCT_inv(block, n = 8):
    ###################################################
    # TODO                                            #
    # DCT_inv 완성                                     #
    # DCT_inv 는 DCT와 다름.                            #
    ###################################################
    def C(w, n):
        result = np.zeros(w.shape)
        (hh, ww) = w.shape
        for a in range(hh):
            for b in range(ww):
                if w[a, b] == 0:
                    result[a, b] = ((1 / n) ** 0.5)
                else:
                    result[a, b] = ((2 / n) ** 0.5)
        return result
    dst = np.zeros(block.shape)
    y, x = dst.shape
    v, u = np.mgrid[0:x, 0:y]
    cv, cu = C(v, n), C(u, n)
    for y_ in range(y):
        for x_ in range(x):
            mask = np.cos((2*x_+1)*u*np.pi/(2*n))*np.cos((2*y_+1)*v*np.pi/(2*n))
            dst[y_, x_] = np.sum(np.sum(block[v, u] * cv * cu * mask))
    return np.round(dst)

def block2img(blocks, src_shape, n = 8):
    ###################################################
    # TODO                                            #
    # block2img 완성                                   #
    # 복구한 block들을 image로 만들기                     #
    ###################################################
    dst = np.zeros(src_shape)
    (h, w) = dst.shape
    x = 0
    for i in range(0, h, n):
        for j in range(0, w, n):
            dst[i:i + n, j:j + n] = blocks[x]
            x = x + 1
    return dst

def Encoding(src, n=8,scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)
    print("block = \n",src[150:158,89:97])


    #subtract 128
    blocks = np.double(blocks)-128
    #blocks -= 128
    b = np.double(src[150:158,89:97])-128
    print("b = \n",b)

    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    # print DCT results
    bd = DCT(b,n=8)
    print("bd = \n",bd)


    #Quantization + thresholding
    Q = Quantization_Luminance(scale_factor)
    QnT = np.round(blocks_dct / Q)
    #print Quantization results
    bq = np.round(bd  / Q)
    print("bq = \n",bq)

    # zigzag scanning
    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_encoding(QnT[i],block_size=n))

    return zz, src.shape, bq

def Decoding(zigzag, src_shape,bq, n=8,scale_factor=1):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_decoding(zigzag[i], block_size=n))

    blocks = np.array(blocks)


    # Denormalizing
    Q = Quantization_Luminance(scale_factor=scale_factor)
    blocks = blocks * Q
    # print results Block * Q
    bq2 = bq * Q
    print("bq2 = \n",bq2)

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    #print IDCT results
    bd2 = DCT_inv(bq2,n=8)
    print("bd2 = \n",bd2)

    # add 128
    blocks_idct += 128

    # print block value
    b2 = bd2 + 128
    print("b2 = \n",b2)

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst, b2



def main():
    scale_factor = 1
    start = time.time()
    # src = cv2.imread('../imgs/Lenna.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.imread('caribou.tif', cv2.IMREAD_GRAYSCALE)

    comp, src_shape,bq = Encoding(src, n=8,scale_factor=scale_factor)
    np.save('comp.npy', comp)
    np.save('src_shape.npy', src_shape)
    # print(comp)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')
    recover_img, b2 = Decoding(comp, src_shape, bq,n=8,scale_factor=scale_factor)
    print("scale_factor : ",scale_factor,"differences between original and reconstructed = \n",src[150:158,89:97]-b2)
    # print(recover_img)
    total_time = time.time() - start
    #
    print('time : ', total_time)
    if total_time > 12:
        print('감점 예정입니다.')
    print(recover_img.shape)
    cv2.imshow('recover img', recover_img/255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
