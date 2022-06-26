import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')

        #up
        pad_img[:p_h, p_w:p_w+w] = src[0, :]
        #down
        pad_img[p_h+h:, p_w:p_w+w] = src[h-1, :]

        #left
        pad_img[:,:p_w] = pad_img[:, p_w:p_w + 1]
        #right
        pad_img[:, p_w+w:] = pad_img[:, p_w+w-1:p_w+w]

    else:
        print('zero padding')

    return pad_img

def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (mask.shape[0]//2, mask.shape[1]//2), pad_type)
    dst = np.zeros((h, w))
    #########################################################
    # TODO                                                  #
    # dst 완성                                              #
    # dst : filtering 결과 image                            #
    #########################################################

    for i in range(h):
        for j in range(w):
            data = sum(sum(mask[:, :] * src_pad[i:i + mask.shape[0], j:j + mask.shape[1]]))
            if data > 255:
                data = 255
            elif data < 0:
                data = 0
            dst[i, j] = data
    dst = (dst+0.5).astype(np.uint8)

    return dst
    
def get_average_mask(fshape):
    print('get average filter')
    ###################################################
    # TODO                                            #
    # mask 완성                                       #
    ###################################################
    mask = np.ones(fshape)/(fshape[0]*fshape[1])
    # mask 확인
    print(mask)
    
    return mask
    
def get_sharpening_mask(fshape):
    print('get sharpening filter')
    ##################################################
    # TODO                                           #
    # mask 완성                                      #
    ##################################################
    zero = np.zeros(fshape)
    zero[fshape[0]//2, fshape[1]//2] = 2
    mask = zero - np.ones(fshape)/(fshape[0]*fshape[1])

    #mask 확인
    print(mask)
    
    return mask


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # 3x3 filter
    dst_average_3x3 = my_filtering(src, get_average_mask(fshape=(3,3)))
    dst_sharpening_3x3 = my_filtering(src, get_sharpening_mask(fshape=(3,3)))

    # 11x13 filter
    dst_average_11x13 = my_filtering(src, get_average_mask(fshape=(11,13)))
    dst_sharpening_11x13 = my_filtering(src, get_sharpening_mask(fshape=(11,13)))

    cv2.imshow('original', src)
    cv2.imshow('average filter 3x3', dst_average_3x3)
    cv2.imshow('sharpening filter 3x3', dst_sharpening_3x3)
    cv2.imshow('average filter 11x13', dst_average_11x13)
    cv2.imshow('sharpening filter 11x13', dst_sharpening_11x13)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
