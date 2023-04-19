# Edge-Detection
## Aim:
To perform edge detection using Sobel, Laplacian, and Canny edge detectors.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
<br>
Import the necessary packages

### Step2:
<br>
Perform edge detection on the picture


### Step4:
<br>Display all the images with edge detection.



 
## Program:

``` Python
# Import the packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image, Convert to grayscale and remove noise

image1=cv2.imread ('T.jpg') 
gray_image = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
plt.title('GRAY IMAGE')
plt.imshow(gray_image,cmap = 'gray')
img = cv2.GaussianBlur(gray_image,(3,3),0)
sobelx = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)
sobelxy =cv2.Sobel(gray_image,cv2.CV_64F,1,1,ksize=5)
plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(gray_image,cmap = 'gray')
plt.title('T1'), plt.xticks([]), plt.yticks([])



# SOBEL EDGE DETECTOR
##sobelx:

plt.subplot(2,2,2)
plt.imshow(sobelx,cmap='gray')
plt.title('sobelx')
plt.xticks([]), plt.yticks([])

# sobely
plt.subplot(2,2,3)
plt.imshow(sobely,cmap='gray')
plt.title('sobely')
plt.xticks([]), plt.yticks([])

# sobelxy
plt.subplot(2,2,4)
plt.imshow(sobelxy,cmap='gray')
plt.title('sobelxy')
plt.xticks([]), plt.yticks([])
plt.show()

# LAPLACIAN EDGE DETECTOR
cv2.waitKey(0)
laplacian = cv2.Laplacian(gray_image,cv2.CV_64F)
plt.imshow(laplacian,cmap='gray')
plt.title('laplacian')
plt.show()


# CANNY EDGE DETECTOR

canny_edges = cv2.Canny(gray_image, 120, 150)
plt.imshow(canny_edges,cmap='gray')
plt.title('canny_edges')
plt.show()


```
## Output:
### SOBEL EDGE DETECTOR
<br>![output](1.png)
<br>
<br>
<br>
<br>
<br>


### LAPLACIAN EDGE DETECTOR
<br> ![output](3.png)
<br>
<br>
<br>
<br>
<br>


### CANNY EDGE DETECTOR
<br> ![output](2.png)
<br>
<br>
<br>
<br>

## Result:
Thus the edges are detected using Sobel, Laplacian, and Canny edge detectors.
