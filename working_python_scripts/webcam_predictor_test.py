
import cv2
import sys
import os
from predictor_images_faces import webcam_predictor

class main():
    def __init__(self, name):
        print(sys.path)
        print(os.path)
        print(os.getcwd())
        print("#######3  hi   ¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
        self.wp = webcam_predictor()

    def predict_thoruch_cam(self):
        faceCascade = cv2.CascadeClassifier("../lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
        video_capture = cv2.VideoCapture(0)
        i=0

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

            # Display the resulting frame
            if ret == False:
                break
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                if (len(faces) > 0):
                    i += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                crop_img = frame[y:y + h, x:x + w]
                crop_img = cv2.resize(crop_img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                name = self.wp.predict(crop_img)
                cv2.putText(frame, name, (x+w, y+h), font, 1, (255, 0, 0))

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

m = main("hello")
m.predict_thoruch_cam()