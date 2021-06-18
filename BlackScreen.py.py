import cv2
import time
import numpy as np

# to save the output in a file -'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# starting the webcam
capture = cv2.VideoCapture(0)

# allowing the webcam to warm up
time.sleep(2)

# capturing background for 60 frames
bg = 0

for i in range(0, 60):
    ret, bg = capture.read()

# fliping the background
bg = np.flip(bg, axis=1)

# reading the caputeblack frame untill the camare is open
while capture.isOpened():
    ret, img = capture.read()

    if not ret:
        break
    
    # fliping the img
    img = np.flip(img, axis=1)

    # converting the color from bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Generating mask to detect black colour

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([104, 153, 80])

    mask_1 = cv2.inRange(hsv,lower_black,upper_black)

    lower_black = np.array([60, 60, 80])
    upper_black = np.array([154, 203, 180])

    mask_2 = cv2.inRange(hsv,lower_black,upper_black)

    mask_1 = mask_1+mask_2


    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))


    mask_2 = cv2.bitwise_not(mask_1)

    res_1 = cv2.bitwise_and(img,img,mask=mask_2)


    res_2 = cv2.bitwise_and(bg,bg,mask=mask_1)


    final_output = cv2.addWeighted(res_1,1,res_2,1,0)
    output_file.write(final_output)

    cv2.imshow("magic", final_output)
    cv2.waitKey(1)


capture.release()
output_file.release()
cv2.destroyAllWindows()

