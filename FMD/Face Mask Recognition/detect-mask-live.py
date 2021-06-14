
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import numpy as np
import cv2


def ppimg(path):
    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img1 = np.array(img)
    img2 = np.array(img).reshape(1, 100, 100, 3)
    return img1, img2


model = tf.keras.models.load_model('Saved Model')

vc = cv2.VideoCapture(0)

while True:
    _, frame = vc.read()
    cv2.imwrite('temp_img.jpg', frame)

    conf = model.predict(ppimg('temp_img.jpg')[1])[0][0]
    text = "Masked" if conf>0.5 else "Unmasked"

    cv2.putText(frame, text, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()

exit(0)
