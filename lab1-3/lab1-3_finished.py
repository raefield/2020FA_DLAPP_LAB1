import cv2
import numpy as np
import matplotlib.pyplot as plot_img

def avgFilter(img):
    # TODO
    img_avg = cv2.blur(img, (3, 3))
    return img_avg

def midFilter(img):
    # TODO
    img_mid = cv2.medianBlur(img, 3)
    return img_mid

def edgeSharpen(img):
    # TODO

    #Sobel_operator = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])  # partial to x
    #Sobel_operator = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])  # partial to y
    #print (Sobel_operator)

    Laplacian_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    print (Laplacian_kernel)

    Sharpen_kernel = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
    print (Sharpen_kernel)

    # Edge detection
    img_edge = cv2.filter2D(img, dst=-1, ddepth=-1, kernel=Laplacian_kernel, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_DEFAULT)

    # Sharpening
    #img_s = cv2.filter2D(img, dst=-1, ddepth=-1, kernel=Sharpen_kernel, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_DEFAULT)
    img_s = cv2.addWeighted(img, 1, img_edge, -0.5, 0)
    img_s[img_s > 255] = 255
    img_s[img_s <0 ] = 0

    return img_edge, img_s



# ------------------ #
#       Denoise      #
# ------------------ #
name1 = '../noise_impulse.png'
name2 = '../noise_gauss.png'
noise_imp = cv2.imread(name1, 0)
noise_gau = cv2.imread(name2, 0)

img_imp_avg = avgFilter(noise_imp)
img_imp_mid = midFilter(noise_imp)
img_gau_avg = avgFilter(noise_gau)
img_gau_mid = midFilter(noise_gau)

cv2.imwrite('img_imp_avg.png', img_imp_avg)
cv2.imwrite('img_imp_mid.png', img_imp_mid)
cv2.imwrite('img_gau_avg.png', img_gau_avg)
cv2.imwrite('img_gau_mid.png', img_gau_mid)

# Use matplotlib to display multiple images
plot_img.figure(1)

plot_img.subplot(2,3,1)
plot_img.imshow(noise_imp, cmap='gray')
plot_img.title('impulse_noise', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

plot_img.subplot(2,3,2)
plot_img.imshow(img_imp_avg, cmap='gray')
plot_img.title('imp_avg', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

plot_img.subplot(2,3,3)
plot_img.imshow(img_imp_mid, cmap='gray')
plot_img.title('imp_mid', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])



plot_img.subplot(2,3,4)
plot_img.imshow(noise_gau, cmap='gray')
plot_img.title('gaussian_noise', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

plot_img.subplot(2,3,5)
plot_img.imshow(img_gau_avg, cmap='gray')
plot_img.title('gau_avg', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

plot_img.subplot(2,3,6)
plot_img.imshow(img_gau_mid, cmap='gray')
plot_img.title('gau_mid', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

#plot_img.show()



# ------------------ #
#       Sharpen      #
# ------------------ #
name = '../mj.tif'
img = cv2.imread(name, 0)

img_edge, img_s = edgeSharpen(img)
cv2.imwrite('mj_edge.png', img_edge)
cv2.imwrite('mj_sharpen.png', img_s)

# Use matplotlib to display multiple images
plot_img.figure(2)

plot_img.subplot(2,2,2)
plot_img.imshow(img, cmap='gray')
plot_img.title('img_original', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

plot_img.subplot(2,2,3)
plot_img.imshow(img_edge, cmap='gray')
plot_img.title('img_edge', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

plot_img.subplot(2,2,4)
plot_img.imshow(img_s, cmap='gray')
plot_img.title('img_sharpening', fontsize=12)
plot_img.xticks([])
plot_img.yticks([])

plot_img.show()