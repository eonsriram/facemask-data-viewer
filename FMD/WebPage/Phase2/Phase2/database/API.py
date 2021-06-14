from sqlops import SQLDB
import datetime
import time
import cv2

# database = "nie"

db = SQLDB()

db.emptyTable()

face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

ctr = 0
while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    except:
        faces = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(faces) > 0 and ctr % 50 == 0:
        login = datetime.datetime.now()
        logout = login + datetime.timedelta(hours=1)

        db.insertRecord("Eon", login, logout)
        print("Inserted")

    cv2.imshow('Image', img)

    ctr += 1

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
