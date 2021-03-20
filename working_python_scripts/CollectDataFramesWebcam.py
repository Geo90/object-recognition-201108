# https://realpython.com/face-detection-in-python-using-a-webcam/

import cv2
import sys
import os

# argv[1] which class to save frames into
label_name = sys.argv[1]

print(os.path)
print(os.getcwd())
faceCascade = cv2.CascadeClassifier("../lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

video_capture = cv2.VideoCapture(0)
i=0
if not os.path.exists('Images/'+label_name):
    os.makedirs('Images/'+label_name)

while True or video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=0
    )

    crop_img = []


    # Display the resulting frame
    if ret == False:
        break
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        if (len(faces) > 0):
            cv2.imwrite('Images/'+label_name+'/'+ label_name + str(i) + '.jpg', frame[y:y + h, x:x + w])
            i += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = frame[y:y + h, x:x + w]

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
