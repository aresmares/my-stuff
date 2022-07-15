import cv2 as cv
import numpy as np

filename = "C:/Users/aresa/Desktop/dev/PathTest/PathTest/example.txt"

file = open(filename)

data = file.readlines()


a = [i.strip().replace(" ","") for i in data]

b = [[int(x)/10 for x in i] for i in a ]

npb = np.array(b)

npb = cv.resize(npb, (500,500), interpolation=cv.INTER_NEAREST)

cv.imshow("image",npb)

cv.waitKey(0) 

cv.destroyAllWindows() 


file.close()
