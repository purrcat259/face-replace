import numpy as np
import cv2

# Reference: https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html

image_path = 'cat.png'
cascade_path = 'haarcascade_frontalface_default.xml'


face_cascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(0)
cat_image = cv2.imread(image_path)
previous_coordinates_and_size = [0, 0, 1, 1]


def get_largest_face(faces):
    if len(faces) == 0:
        return None
    largest_face = None
    largest_area = 0
    for (x, y, w, h) in faces:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)
    return largest_face


while True:
    ret, img = cap.read()
    # cv2.imshow('frame', img)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        grey_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(125, 125)
    )
    # print('Found {} faces'.format(len(faces)))
    largest_face = get_largest_face(faces)
    if largest_face:
        x, y, w, h = largest_face
        # print(largest_face)
        # Resize cat image to rectangle size
        current_frame_cat_image = cv2.resize(cat_image, (w, h), interpolation=cv2.INTER_CUBIC)
        img[y : y + h, x : x + w, :] = current_frame_cat_image
        previous_coordinates_and_size[0] = x
        previous_coordinates_and_size[1] = y
        previous_coordinates_and_size[2] = w
        previous_coordinates_and_size[3] = h
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # cv2.rectangle(img, (x, y), (x + 1, y + 1), (0, 255, 0), 2)
    else:
        # Persist image till the next face detection to avoid blinking
        x, y, w, h = previous_coordinates_and_size
        current_frame_cat_image = cv2.resize(cat_image, (w, h), interpolation=cv2.INTER_CUBIC)
        img[y: y + h, x: x + w, :] = current_frame_cat_image
    cv2.imshow('Face Replace', img)
    if cv2.waitKey(1) == 27:  # Escape key
        break

cap.release()
cv2.destroyAllWindows()
