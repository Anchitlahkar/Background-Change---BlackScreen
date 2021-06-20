from typing import final
import cv2
import time
import numpy as np
from PIL import Image

# to save the output in a file -'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 5, (640,480))

capture = cv2.VideoCapture(0)

image = Image.open("bangkok.jpg")
print(image)

time.sleep(2)

# reading the caputeblack frame untill the camare is open
while capture.isOpened():
    ret, frame = capture.read()

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(33) == 27:
        break

    frame = cv2.resize(frame, (640,480))
    image = image.resize((640,480))

    
    frame = np.flip(frame, axis=1)

    # converting the color from bgr to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_black = np.array([0, 0, 0])
    upper_black = np.array([94, 133, 60])

    mask = cv2.inRange(frame,lower_black,upper_black)


    res = cv2.bitwise_and(frame,frame,mask=mask)


    final_output = frame - res
    final_output = np.where(final_output==0,image , final_output)

    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)


capture.release()
frame.release()
cv2.destroyAllWindows()

