import cv2
import numpy as np
import math

def splitRGB(img):
    # TODO
    #(B_map, G_map, R_map) = cv2.split(img)  # color sequence in opencv: B, G, R
    (B, G, R) = cv2.split(img)  # color sequence in opencv: B, G, R

    zeros = np.zeros(img.shape[:2], dtype = "uint8")

    B_map = cv2.merge([B, zeros, zeros])
    G_map = cv2.merge([zeros, G, zeros])
    R_map = cv2.merge([zeros, zeros, R])

    return R_map, G_map, B_map


def splitHSV(img):
    # TODO
    (B, G, R) = cv2.split(img)  # color sequence in opencv: B, G, R
    #(H, S, V) = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #(H_map, S_map, V_map) = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    zeros = np.zeros(img.shape[:2], dtype = "uint8")

    B_map = cv2.merge([B, zeros, zeros])
    G_map = cv2.merge([zeros, G, zeros])
    R_map = cv2.merge([zeros, zeros, R])

    H_map = cv2.cvtColor(B_map, cv2.COLOR_BGR2HSV)
    S_map = cv2.cvtColor(G_map, cv2.COLOR_BGR2HSV)
    V_map = cv2.cvtColor(R_map, cv2.COLOR_BGR2HSV)

    return H_map, S_map, V_map



# Bilinear-interpolation Ref. : https://stackoverflow.com/questions/26142288/resize-an-image-with-bilinear-interpolation-without-imresize/26143655
#                               https://eng.aurelienpierre.com/2020/03/bilinear-interpolation-on-images-stored-as-python-numpy-ndarray/

def resize(img, size):

    height_in, width_in, img_channel_in = img.shape
    height_out = int(size * height_in)
    width_out = int(size * width_in)  
    img_t = np.zeros((height_out, width_out, 3), np.uint8)

    for i in range(height_out):
        for j in range(width_out):
            # Relative coordinates of the pixel in output space
            x_out = j // width_out
            y_out = i // height_out
 
            # Corresponding absolute coordinates of the pixel in input space
            x_in = x_out * width_in
            y_in = y_out * height_in
 
            # Nearest neighbours coordinates in input space
            x_prev = int(np.floor(x_in))
            x_next = x_prev + 1
            y_prev = int(np.floor(y_in))
            y_next = y_prev + 1
 
            # Sanitize bounds - no need to check for < 0
            #x_prev = min(x_prev, width_in - 1)
            #x_next = min(x_next, width_in - 1)
            #y_prev = min(y_prev, height_in - 1)
            #y_next = min(y_next, height_in - 1)
            
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






'''
class MotionDetect(object):
    """docstring for MotionDetect"""
    def __init__(self, shape):
        super(MotionDetect, self).__init__()

        self.shape = shape
        self.avg_map = np.zeros((self.shape[0], self.shape[1]), dtype='float')
        self.alpha = 0.8 # you can ajust your value
        self.threshold = 40 # you can ajust your value

        print("MotionDetect init with shape {}".format(self.shape))

    def getMotion(self, img):
        assert img.shape == self.shape, "Input image shape must be {}, but get {}".format(self.shape, img.shape)

        # Extract motion part (hint: motion part mask = difference between image and avg > threshold)
        # TODO

        # Mask out unmotion part (hint: set the unmotion part to 0 with mask)
        # TODO

        # Update avg_map
        # TODO

        return motion_map
'''


# ------------------ #
#     RGB & HSV      #
# ------------------ #
name = "../data.png"
img = cv2.imread(name)
if img is not None:
    print("Reading {} success. Image shape {}".format(name, img.shape))
else:
    print("Faild to read {}.".format(name))

R_map, G_map, B_map = splitRGB(img)
H_map, S_map, V_map = splitHSV(img)

cv2.imwrite('data_R.png', R_map)
cv2.imwrite('data_G.png', G_map)
cv2.imwrite('data_B.png', B_map)

cv2.imwrite('data_H.png', H_map)
cv2.imwrite('data_S.png', S_map)
cv2.imwrite('data_V.png', V_map)


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
img_big = resize(img, 2)
img_small = resize(img, 0.5)
img_big_cv = cv2.resize(img, (width*2, height*2))
img_small_cv = cv2.resize(img, (width//2, height//2))

cv2.imwrite('data_2x.png', img_big)
cv2.imwrite('data_0.5x.png', img_small)
cv2.imwrite('data_2x_cv.png', img_big_cv)
cv2.imwrite('data_0.5x_cv.png', img_small_cv)




'''
# ------------------ #
#  Video Read/Write  #
# ------------------ #
name = "../data.mp4"
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

        # Write 1 frame to output video
        out.write(motion_map)
    else:
        break

# Release resource
cap.release()
out.release()

'''
