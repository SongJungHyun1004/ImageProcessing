import cv2
import numpy as np

def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################
    (h, w) = B.shape
    (p_h, p_w) = (S.shape[0] // 2, S.shape[1] // 2)
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            if B[row, col] == 1:
                pad_img[row:row + S.shape[0], col:col + S.shape[1]] = S
    dst = pad_img[p_h:p_h + h, p_w:p_w + w]
    return dst

def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    (h, w) = B.shape
    (p_h, p_w) = (S.shape[0] // 2, S.shape[1] // 2)
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = B
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            if B[row, col] == 1:
                if pad_img[row:row + S.shape[0], col:col + S.shape[1]].all() == S.all():
                    dst[row, col] = 1
    return dst

def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################
    dst = dilation(erosion(B, S), S)
    return dst

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    dst = erosion(dilation(B, S), S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('test_binary_image.png', (B * 255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('test_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('test_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('test_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('test_closing.png', img_closing)

    test_img = cv2.imread("morphology_img.png", cv2.IMREAD_GRAYSCALE)
    test_img = test_img / 255.
    test_img = np.round(test_img)
    test_img = test_img.astype(np.uint8)

    img_dilation = dilation(test_img, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(test_img, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(test_img, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(test_img, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)




