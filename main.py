import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

cap = cv2.VideoCapture(0)
idx = 0
orb = cv2.ORB_create()


def hough(frame):

    edges = cv2.Canny(frame, 100, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, 100, 10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            res = cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame


def region_interet(frame):
    kp = orb.detect(frame, None)
    res = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0), flags=0)
    return res


def transforee_affine(frame):
    s = 0.5
    teta = np.pi
    alpha = s*np.cos(teta)
    beta = s*np.sin(teta)
    cx = frame.shape[0]//2
    cy = frame.shape[1]//2
    tx = 10
    ty = 15
    M = np.float32([[alpha, beta,  (1-alpha)*cx-beta*cy+tx],
                    [-beta, alpha, beta*cx+(1-alpha)*cy+ty]])

    row = frame.shape[0]
    col = frame.shape[1]
    res = cv2.warpAffine(frame, M, (col, row))
    return res


def transformee_non_lineaire(frame):
    x = np.arange(frame.shape[1], dtype=int)
    y = np.arange(frame.shape[0], dtype=int)
    xv, xy = np.meshgrid(x, y)
    A = 30
    xvtil = xv - frame.shape[1]/2
    xytil = xy - frame.shape[0]/2
    tetax = 100000
    tetay = 100000
    alpha = xvtil + xytil
    dx = A * np.cos(alpha) * np.exp(-((xvtil ** 2 / tetax) + (xytil ** 2 / tetay)))
    dy = A * np.sin(alpha) * np.exp(-((xvtil ** 2 / tetax) + (xytil ** 2 / tetay)))
    #dx = xv*np.cos(xv)
    #dy = xy*np.cos(xy)
    mapx = xv + dx
    mapy = xy + dy
    res = cv2.remap(frame, np.float32(mapx), np.float32(mapy), cv2.INTER_NEAREST)
    return res


while True:
    ret, frame = cap.read()  # 1 frame acquise à chaque iteration

    frame = transformee_non_lineaire(frame)

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
