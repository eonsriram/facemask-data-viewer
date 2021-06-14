import face_recognition as fr
import cv2
import numpy as np
import os
import tensorflow as tf
from sqlops import SQLDB
import datetime

db = SQLDB(password="Aditya@123")
db.emptyTable()

known_face_encodings = []
known_face_names = []

for img in os.listdir('known_faces'):
    image = fr.load_image_file("known_faces/" + img)

    encoding = fr.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(img.split(".")[0])


def ppimg(path):
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    # img1 = np.array(img)
    img2 = np.array(img).reshape(1, 100, 100, 3)
    return img2


# model = tf.keras.models.load_model('Saved Model')


video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
ctr = 0
while True:

    ret, frame = video_capture.read()
    cv2.imwrite('temp_img.jpg', frame)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = fr.face_locations(rgb_small_frame)
        print(len(known_face_names))
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            try:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    x = known_face_names[best_match_index].split("_")
                    name = x[0]
                    flag = 0 if x[1] == "nomask" else 1
                    # conf = model.predict(ppimg('temp_img.jpg'))[0][0]
                    # text = "Masked" if conf > 0.5 else "Unmasked"
                    # full_name = name + text
                    if ctr % 5 == 0:
                        db.insertRecord(name=x[0], login=datetime.datetime.now(), inmask=str(flag))
                    ctr += 1
            except:
                name = "Unknown"

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        color = (0, 255, 0)  # if text == "Masked" else (0,0,255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
