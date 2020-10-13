import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def splitRGB(img):
    # TODO
    #(B_map, G_map, R_map) = cv2.split(img)  # color sequence in opencv: B, G, R
    (B, G, R) = cv2.split(img)  # color sequence in opencv: B, G, R

    zeros = np.zeros(img.shape[:2], dtype = "uint8")

    B_map = cv2.merge([zeros, zeros, B])
    G_map = cv2.merge([zeros, G, zeros])
    R_map = cv2.merge([R, zeros, zeros])

    return B_map, G_map, R_map


def splitHSV(img):
    # TODO

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (H_map, S_map, V_map) = cv2.split(img_hsv)

    return H_map, S_map, V_map



# Bilinear-interpolation Ref. : https://eng.aurelienpierre.com/2020/03/bilinear-interpolation-on-images-stored-as-python-numpy-ndarray/

def resize(img, size):

    height_in, width_in, img_channel_in = img.shape
    height_out = int(size * height_in)
    width_out = int(size * width_in)  
    img_t = np.zeros((height_out, width_out, 3), np.uint8)

    for i in range(height_out):
        for j in range(width_out):
            # Relative coordinates of the pixel in output space
            x_out = j / width_out
            y_out = i / height_out
 
            # Corresponding absolute coordinates of the pixel in input space
            x_in = x_out * width_in
            y_in = y_out * height_in
 
            # Nearest neighbours coordinates in input space
            x_prev = int(np.floor(x_in))
            x_next = x_prev + 1
            y_prev = int(np.floor(y_in))
            y_next = y_prev + 1
 
            # Sanitize bounds - no need to check for < 0
            x_prev = min(x_prev, width_in - 1)
            x_next = min(x_next, width_in - 1)
            y_prev = min(y_prev, height_in - 1)
            y_next = min(y_next, height_in - 1)
            
            # Distances between neighbour nodes in input space
            Dy_next = y_next - y_in
            Dy_prev = 1 - Dy_next # because next - prev = 1
            Dx_next = x_next - x_in
            Dx_prev = 1 - Dx_next # because next - prev = 1
            
            # Interpolate over 3 RGB layers
            for c in range(3):
                img_t[i][j][c] = Dy_prev * (img[y_next][x_prev][c] * Dx_next + img[y_next][x_next][c] * Dx_prev) \
                + Dy_next * (img[y_prev][x_prev][c] * Dx_next + img[y_prev][x_next][c] * Dx_prev)

    return img_t







class MotionDetect(object):
    """docstring for MotionDetect"""
    def __init__(self, shape):
        super(MotionDetect, self).__init__()

        self.shape = shape
        self.avg_map = np.zeros((self.shape[0], self.shape[1]), dtype='float')
        self.alpha = 0.8 # you can adjust your value
        self.threshold = 100 # you can adjust your value
        
        self.counter = 0
        self.Avg_frame = 0
        print("MotionDetect init with shape {}".format(self.shape))

    def getMotion(self, img):
        assert img.shape == self.shape, "Input image shape must be {}, but get {}".format(self.shape, img.shape)

        # Hint:
        # Average all frames to simulate the scene without moving object
        #  – Motion = Image - Avg_map
        #  – Avg_map = Avg_map*alpha + Image*(1-alpha)

        counter = self.counter + 1
        Avg_frame = self.Avg_frame
        avg_map = self.avg_map
        print ('----------------------------------------------------->  Start counter = ', counter)
        #Avg_frame = self.avg_map

        while (counter != 0):
        # Extract motion part (hint: motion part mask = difference between image and avg > threshold)
        # TODO
            #(B_map, G_map, R_map) = cv2.split(img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print ('img_gray = ')
            print (img_gray)

            Avg_frame_update = cv2.mean(img_gray)[0]

            Avg_frame = int((Avg_frame_update + Avg_frame * counter) / (counter + 1))
            avg_height, avg_width, avg_channel = img.shape
            print ('--------------------->  Accumulate count = ', counter)
            print (Avg_frame)
            
            avg_map = np.full((avg_height, avg_width), Avg_frame, dtype=np.uint8) 
            #avg_map = np.ones((avg_height, avg_width)) * Avg_frame
            print ('avg_map = ')
            print (avg_map)
            
            diff = np.full((avg_height, avg_width), 0, dtype=np.uint8) 
            #diff = abs(img_gray - avg_map)
            diff = cv2.subtract(img_gray, avg_map, dst=None, mask=None, dtype=None)
            for i in range(avg_height):
                for j in range(avg_width):

                    if (diff[i][j] > self.threshold):
                        diff[i][j] = 1
                    else:
                        diff[i][j] = 0
            
            print ('diff = ')
            print (diff)


            # Mask out unmotion part (hint: set the unmotion part to 0 with mask)
             # TODO
            motion_map = img_gray * diff
            print ('motion_map = ')
            print (motion_map)

        # Update avg_map
        # TODO

            counter = counter +1

            return motion_map



# ------------------ #
#     RGB & HSV      #
# ------------------ #
name = "../data.png"
img = cv2.imread(name)
if img is not None:
    print("Reading {} success. Image shape {}".format(name, img.shape))
else:
    print("Faild to read {}.".format(name))

B_map, G_map, R_map = splitRGB(img)
H_map, S_map, V_map = splitHSV(img)

cv2.imwrite('data_B.png', R_map)
cv2.imwrite('data_G.png', G_map)
cv2.imwrite('data_R.png', B_map)

cv2.imwrite('data_H.png', H_map)
cv2.imwrite('data_S.png', S_map)
cv2.imwrite('data_V.png', V_map)
'''
# Use matplotlib to display multiple images
plt.figure(1, figsize=(18, 8))

plt.subplot(2,3,1)
plt.imshow(R_map)
plt.title('R_map.png', fontsize=12)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.imshow(G_map)
plt.title('G_map.png', fontsize=12)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.imshow(B_map)
plt.title('B_map.png', fontsize=12)
plt.xticks([])
plt.yticks([])



plt.subplot(2,3,4)
plt.imshow(H_map, cmap='gray')
plt.title('H_map.png', fontsize=12)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,5)
plt.imshow(S_map, cmap='gray')
plt.title('S_map.png', fontsize=12)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,6)
plt.imshow(V_map, cmap='gray')
plt.title('V_map.png', fontsize=12)
plt.xticks([])
plt.yticks([])
'''


# ------------------ #
#   Interpolation    #
# ------------------ #
name = "../data.png"
img = cv2.imread(name)
if img is not None:
    print("Reading {} success. Image shape {}".format(name, img.shape))
else:
    print("Faild to read {}.".format(name))

height, width, channel = img.shape
#img_big = resize(img, 2)
#img_small = resize(img, 0.5)
img_big_cv = cv2.resize(img, (width*2, height*2))
img_small_cv = cv2.resize(img, (width//2, height//2))

#cv2.imwrite('data_2x.png', img_big)
#cv2.imwrite('data_0.5x.png', img_small)
cv2.imwrite('data_2x_cv.png', img_big_cv)
cv2.imwrite('data_0.5x_cv.png', img_small_cv)


'''
fig2, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.5), dpi=180, sharex=True, sharey=True)
ax[2].imshow(img_small)
ax[2].set_title('data_2x.png', fontsize=12)
ax[1].imshow(img)
ax[1].set_title('original.png', fontsize=12)
ax[0].imshow(img_big)
ax[0].set_title('data_0.5x.png', fontsize=12)

fig3_cv, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.5), dpi=180, sharex=True, sharey=True)
ax[2].imshow(img_small_cv)
ax[2].set_title('data_2x_cv.png', fontsize=12)
ax[1].imshow(img)
ax[1].set_title('original.png', fontsize=12)
ax[0].imshow(img_big_cv)
ax[0].set_title('data_0.5x_cv.png', fontsize=12)
'''
#plt.show()






# ------------------ #
#  Video Read/Write  #
# ------------------ #
#name = "../data.mp4"
name = "../data_cut.mp4"
# Input reader
cap = cv2.VideoCapture(name)
fps = cap.get(cv2.CAP_PROP_FPS)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, fps, (w, h), True)

# Motion detector
mt = MotionDetect(shape=(h,w,3))

# Read video frame by frame
while True:
    # Get 1 frame
    success, frame = cap.read()

    if success:
        motion_map = mt.getMotion(frame)
        #plt.imshow(motion_map)
        # Write 1 frame to output video
        out.write(motion_map)
    else:
        break

# Release resource
cap.release()
out.release()


