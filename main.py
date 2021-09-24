import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

cap = cv2.VideoCapture(0)

idx = 0

while True:
    ret, frame = cap.read()  # 1 frame acquise à chaque iteration

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cv2.imshow('Capture_Video', frame)  # affichage
    key = cv2.waitKey(1)  # on évalue la touche pressée

    if key & 0xFF == ord('q'):  # si appui sur 'q'
        break  # sortie de la boucle while

    elif key & 0xFF == ord('c'):  # si appuie sur 'c'
        cv2.imwrite(f"img_{idx}.jpg", frame)  # on enregistre l'image avec un idx pour en faire plusieurs
        idx += 1  # on incremente l'index
        time.sleep(0.5)  # on met une petite pause parce que sinon cela prend plusieurs image si on appuie trop

    channels = [0]
    mask = None
    histSize = [256]
    ranges = [0, 256]
    hist = cv2.calcHist(frame, channels, mask, histSize, ranges)

    plt.plot(hist)  # où hist est la sortie de cv2.calcHist
    plt.title('Histogramme')
    plt.draw()  # execute l'affichage
    plt.pause(0.0001)  # delai nécessaire a l'affichage
    plt.cla()  # évite la superposition des courbes

cap.release()
cv2.destroyAllWindows()
