import numpy as np
import cv2
import sys
import time

threshold = 15

def euclideanDistance(origFrame, destFrame):   
    diff = np.float32(origFrame) - np.float32(destFrame)
    norm32 = np.sqrt(diff[:,:,0]**2 + diff[:,:,1]**2 + diff[:,:,2]**2) / np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32 * 255)
    return dist

cv2.namedWindow('diference')
camera = cv2.VideoCapture(0)

_, oldFrame = camera.read()
rows, cols, _ = np.shape(oldFrame)

facecount = 0
while(True):
    _, newFrame = camera.read()     
    dist = euclideanDistance(oldFrame, newFrame)
   
    mod = cv2.GaussianBlur(dist, (9,9), 0)  
    _, thresh = cv2.threshold(mod, 100, 255, 0)
    _, stDev = cv2.meanStdDev(mod)

    cv2.imshow('diference', mod)
    if stDev > threshold:
        print("Detection " + str(time.asctime()));
        sys.stdout.flush();
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

    oldFrame = newFrame

camera.release()
cv2.destroyAllWindows()
