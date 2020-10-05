# DLAPP_LAB1

## 1-1 Get into OpenCV
 - **RGB & HSV (10%) [Done!]** 
   
   Read the *data.png* and do: 

    (1) Split the color image to Red/Green/Blue image

    (2) Convert the RGB image to HSV image and show each channel

 - **Interpolation (10%)**
   
   Implement a resize function that resize the *data.png* to **2x bigger** and **2x smaller** using *Bilinear Interpolation*.

     - Compare *cv2.resize()* and your *resize()* function

 - **Video Read/Write (10%)**

    (1) Read the *data.mp4*, try to get the moving objects in the video

    (2) Write the new video *output.avi* that only have moving objects

     * *Hint*:
        - Average all frames to simulate the scene without moving object
        - Motion = Image - Avg_map
        - Avg_map = Avg_map * alpha + Image * (1-alpha)



## 1-2 Enhancement [Done!]
 - **Gamma Correction (10%)**
 
    (1) Get 1 image from *data.mp4*

    (2) Enhance the contrast of both bright part and dark part with different gamma

 - **Histogram Equalization (20%)**

    (1) Read the *hist.png*, try to implement Histogram Equalization and show the result image and histogram

    (2) Compare with *cv2.equalizeHist()*



## 1-3 Mask Processing [Done!]
 - **Denoise (20%)**
     - Gaussian noise & Salt and Paper noise
     - Average filters & Medium filters
    
    (1) Try *3x3 average filter* & *3x3 medium filter* on Gaussian noise & Salt and Paper noise image.

    (2) Compare 4 results.

 - **Sharpen (20%)**
     - Edge detection

    (1) Try to find the edge of Michael Jackson using *laplacian filter*: [[0, 1, 0],[1, -4, 1],[0, 1, 0]]

    (2) Sharpen the image with edge map.



## Bonus (10%)
 - **Use the technique above to make 1-1 Video Read/Write perform better.**
     - Less noise
     - More complete region


