import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # 1 frame acquise à chaque iteration
    cv2.imshow('Capture_Video', frame)  # affichage
    key = cv2.waitKey(1)  # on évalue la touche pressée
    if key & 0xFF == ord('q'):  # si appui sur 'q'
        break  # sortie de la boucle while
cap.release()
cv2.destroyAllWindows()
