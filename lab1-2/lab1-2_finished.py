import cv2
import numpy as np
import matplotlib.pyplot as plt

def gammaCorrection(img, gamma=1.0):

    #img_g = np.power(img/float(np.max(img)), gamma)
    
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    # TODO
    img_rescaled = img/float(np.max(img))

    # Apply gamma correction using the lookup table
    # TODO
    img_g = np.power(img_rescaled, gamma)
    
    return img_g


# Histogram Equalization Ref.: https://www.itread01.com/content/1540993224.html
def histEq(gray):
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist / gray.size
    print(hist)

    # Convert the histogram to Cumulative Distribution Function (CDF)
    # TODO
    histogram = np.bincount(gray.flatten(), minlength=256)
    cdf = np.cumsum(histogram)
    print(cdf)
    # print(histogram / gray.size) = print(hist)

    # Build a lookup table mapping the pixel values [0, 255] to their new grayscale value
    # TODO
    uniform_hist = 255 * (cdf / (gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')

    # Apply histogram equalization using the lookup table
    # TODO
    height, width = gray.shape
    img_h = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            img_h[i,j] = uniform_hist[gray[i,j]]

    return img_h





# ------------------ #
#  Gamma Correction  #
# ------------------ #
name = "../data.mp4"
cap = cv2.VideoCapture(name)
success, frame = cap.read()
if success:
    print("Success reading 1 frame from {}".format(name))
else:
    print("Faild to read 1 frame from {}".format(name))
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img_g1 = gammaCorrection(gray, 0.5)
img_g2 = gammaCorrection(gray, 2)
cv2.imwrite('gray.png', gray)
cv2.imwrite('data_g0.5.png', img_g1*255)    # img_g1*255: Modify the range from 0-1 to 0-255
cv2.imwrite('data_g2.png', img_g2*255)      # img_g2*255: Modify the range from 0-1 to 0-255

# Use matplotlib to display multiple images
plt.figure(1, figsize=(18, 6))

plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title('gray.png', fontsize=12)
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(img_g1, cmap='gray')
plt.title('data_g0.5.png', fontsize=12)
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(img_g2, cmap='gray')
plt.title('data_g2.png', fontsize=12)
plt.xticks([])
plt.yticks([])




# ------------------------ #
#  Histogram Equalization  #
# ------------------------ #
name = "../hist.png"
img = cv2.imread(name, 0)

img_h = histEq(img)
img_h_cv = cv2.equalizeHist(img)
cv2.imwrite("hist_h.png", img_h)
cv2.imwrite("hist_h_cv.png", img_h_cv)

# save histogram
plt.figure(2, figsize=(18, 6))
plt.subplot(1,3,1)
plt.bar(range(1,257), cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(-1))
plt.subplot(1,3,2)
plt.bar(range(1,257), cv2.calcHist([img_h], [0], None, [256], [0, 256]).reshape(-1))
plt.subplot(1,3,3)
plt.bar(range(1,257), cv2.calcHist([img_h_cv], [0], None, [256], [0, 256]).reshape(-1))
plt.savefig('hist_plot.png')

plt.show()