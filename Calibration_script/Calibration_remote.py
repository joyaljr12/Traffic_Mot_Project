import cv2

file = open("calibValues.py", "w")
frame = cv2.imread('traffic.png')

bbox_d = []
ndir = input("Please enter the number of direction:\n")
dir = []

for i in range(int(ndir)):
    bbox_d.append(cv2.selectROI("dir", frame, False))
    dir.append(input("Please enter the direction name:\n"))

bbox_d_w = repr(bbox_d)
dir_w = repr(dir)
file.write("bbox_d = " + bbox_d_w + "\n")
file.write("dir = " + dir_w + "\n")
file.close()



            
