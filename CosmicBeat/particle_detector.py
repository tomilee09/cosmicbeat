import cv2
import numpy as np
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play

sonido = AudioSegment.from_file(file = "fa.mp3", format = "mp3")
# play(sonido)

cap = cv2.VideoCapture('cloud.mp4')
start_time = 550  # tiempo de inicio en segundos # puros rayos fuertes
# start_time = 120  # tiempo de inicio en segundos # radiacion
# start_time = 620  # tiempo de inicio en segundos
cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=10,detectShadows=False)


while True:
    ret, frame = cap.read()
    if ret == False: break
    # transformo esto frame en grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # le aplico un filtro para eliminar ruido
    kernel = np.ones((5, 5), np.float32)/36
    blur = cv2.filter2D(gray, -1, kernel)
    
    # otro filtro de blur para eliminar ruido
    erosion = cv2.erode(blur, kernel, iterations=1)

    blur = cv2.GaussianBlur(erosion, (5, 5), 0)

    # mascara para eliminar fondo, notese que está ahora en blanco o negro (binario)
    fgmask = fgbg.apply(blur)

    # mask = cv2. blur (fgmask , (15 , 15))
    # ret , mask = cv2. threshold (mask , 64 , 255 , cv2. THRESH_BINARY )

    # encuentro los bordes
    contornos,hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # dibujo los contornos en fgmask
    largos = []
    for i in range(len(contornos)):
        cv2.drawContours(frame, contornos, i, (0,0,i), 5)
        x_min = 1000000
        x_max = 0
        y_min = 1000000
        y_max = 0
        for j in range(len(contornos[i])):
            if x_min > contornos[i][j][0][0]:
                x_min = contornos[i][j][0][0]
            
            if x_max < contornos[i][j][0][0]:
                x_max = contornos[i][j][0][0]
            
            if y_min > contornos[i][j][0][1]:
                y_min = contornos[i][j][0][0]
            
            if y_max < contornos[i][j][0][1]:
                y_max = contornos[i][j][0][0]
            
        largo = ((x_max - x_min)**2 + (y_max - y_min)**2)**(0.5)
        largos.append(largo)
    for elemento in largos:
        if elemento > 300:
            play(sonido)
            break

    # muestro en pantalla la imagen sin fondo y la imagen con blur
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('frame', frame)
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        # print(contornos[0][0])
        # print(largos)
        break

cap.release()
cv2.destroyAllWindows()