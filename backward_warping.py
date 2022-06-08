import cv2
import numpy as np


def backward_warping(M, img):
    ###################################################################
    # TODO                                                            #
    # backward 방식으로 warping 수행                                     #
    # 1. M의 역행렬은 numpy를 사용해 구함                                  #
    # 2. dst의 각 픽셀마다 원본의 좌표를 계산                               #
    # 원본 좌표가 소숫점이 나올 확률이 높기 떄문에 bilinear interpolation 수행 #
    ###################################################################
    src_h, src_w = img.shape
    y_scale = M[1, 1]
    x_scale = M[0, 0]

    dst_h = max(int(src_h*y_scale+0.5), src_h)
    dst_w = max(int(src_w*x_scale+0.5), src_w)
    dst = np.zeros((dst_h, dst_w), img.dtype)

    inv_M = np.linalg.inv(M)

    for y in range(dst_h):
        for x in range(dst_w):
            dst_coordinate = inv_M @ np.array([x, y, 1])
            x_, y_ = dst_coordinate[0], dst_coordinate[1]
            if x_ < 0 or x_ > dst_w - 1 or y_ < 0 or y_ > dst_h - 1:
                dst[y, x] = 0
            else:
                m = min(int(y_), src_h - 2)
                n = min(int(x_), src_w - 2)
                t = y_ - int(y_)
                s = x_ - int(x_)
                dst[y, x] = (1 - s) * (1 - t) * img[m, n] \
                            + s * (1 - t) * img[m, n + 1] \
                            + (1 - s) * t * img[m + 1, n] \
                            + s * t * img[m + 1, n + 1]

    return dst


if __name__ == '__main__':
    img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

    M1 = np.array([[1, 0, 50],
                   [0, 1, 100],
                   [0, 0, 1]])

    M2 = np.array([[1.5, 0, 0],
                   [0, 1.5, 0],
                   [0, 0, 1]])

    angle = np.deg2rad(15)
    M3 = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])

    M4 = np.array([[1, 0.2, 0],
                   [0.2, 1, 0],
                   [0, 0, 1]])

    dst1 = backward_warping(M1, img)
    dst2 = backward_warping(M2, img)
    dst3 = backward_warping(M3, img)
    dst4 = backward_warping(M4, img)

    cv2.imshow('original', img)
    cv2.imshow('translation', dst1)
    cv2.imshow('scaling', dst2)
    cv2.imshow('rotation', dst3)
    cv2.imshow('shear', dst4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

