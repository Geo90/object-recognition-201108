# https://realpython.com/face-recognition-with-python/

import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

print(imagePath)
print(cascPath)

if hasattr(cv2, 'data'):
    print('Cascades are here:', cv2.data.haarcascades)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("lib/python3.6/site-packages/cv2/data/haarcascade_frontalcatface.xml")

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=2,
    minSize=(65, 65),
    flags=0
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
