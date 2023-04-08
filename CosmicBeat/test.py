import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
#     cv2.imshow('webcam', frame)
# # press escape to exit
#     if (cv2.waitKey(30) == 27):
#        break
# cap.release()
# cv2.destroyAllWindows()

##############################################################

# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
#     if lines is not None:
#         for line in lines:
#             rho, theta = line[0]
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a*rho
#             y0 = b*rho
#             x1 = int(x0 + 1000*(-b))
#             y1 = int(y0 + 1000*(a))
#             x2 = int(x0 - 1000*(-b))
#             y2 = int(y0 - 1000*(a))
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

##########################################################

# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

##############################################################################

# import cv2
# import numpy as np

# # Inicializar la cámara
# cap = cv2.VideoCapture(0)

# # Configurar los parámetros para la segmentación de imágenes
# low_threshold = 50
# high_threshold = 150
# kernel_size = 5

# # Configurar los parámetros para la detección de bordes
# minLineLength = 100
# maxLineGap = 10

# # Inicializar la lista de puntos de trayectoria
# trajectory = []

# while True:
#     # Capturar un fotograma de la cámara
#     ret, frame = cap.read()

#     # Convertir el fotograma a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Aplicar la segmentación de imágenes y la detección de bordes
#     edges = cv2.Canny(cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0), low_threshold, high_threshold)
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)

#     # Dibujar las líneas detectadas en el fotograma
#     if lines is not None:
#         for x1, y1, x2, y2 in lines[0]:
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             # Agregar los puntos de trayectoria a la lista
#             trajectory.append((x1, y1))

#     # Mostrar el fotograma y la trayectoria en pantalla
#     cv2.imshow('frame', frame)
#     if len(trajectory) > 1:
#         for i in range(len(trajectory)-1):
#             cv2.line(frame, trajectory[i], trajectory[i+1], (255, 0, 0), 2)
#         cv2.imshow('trajectory', frame)

#     # Salir si se presiona la tecla 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Liberar la cámara y cerrar las ventanas
# cap.release()
# cv2.destroyAllWindows()

####################################################################3

import cv2
import numpy as np

# Inicializar la cámara
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('cloud.mp4')
start_time = 120  # tiempo de inicio en segundos
cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

# Configurar los parámetros para la segmentación de imágenes
low_threshold = 50
high_threshold = 200
kernel_size = 15

# Configurar los parámetros para la detección de bordes
minLineLength = 100
maxLineGap = 30

# Configurar los parámetros para la separación de trazas
max_distance = 150
max_age = 50

# Inicializar la lista de partículas y trazas
particles = []
tracks = []

# Definir una función para agregar una partícula a la lista de partículas
def add_particle(x, y):
    particles.append({'id': len(particles), 'x': x, 'y': y, 'age': 0})

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar la segmentación de imágenes y la detección de bordes
    edges = cv2.Canny(cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0), low_threshold, high_threshold)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)

    # Dibujar las líneas detectadas en el fotograma y agregar las partículas a la lista
    if lines is not None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            add_particle((x1 + x2) // 2, (y1 + y2) // 2)

    # Actualizar las trazas existentes y crear nuevas trazas
    for track in tracks:
        track['age'] += 1
        track['points'].append((track['x'], track['y']))
        for particle in particles:
            if np.sqrt((track['x'] - particle['x'])**2 + (track['y'] - particle['y'])**2) < max_distance:
                track['x'] = particle['x']
                track['y'] = particle['y']
                track['age'] = 0
                particles.remove(particle)
                break
        else:
            if track['age'] > max_age:
                tracks.remove(track)

    for particle in particles:
        tracks.append({'id': particle['id'], 'x': particle['x'], 'y': particle['y'], 'age': 0, 'points': [(particle['x'], particle['y'])]})
        particles.remove(particle)

    # Dibujar las trazas en el fotograma
    for track in tracks:
        cv2.polylines(frame, [np.array(track['points'], np.int32)], False, (255, 0, 0), 2)

    # Mostrar el fotograma con las líneas y las trazas
    cv2.imshow('frame', frame)

    # Esperar por la tecla 'q' para salir del bucle principal
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()

##################################################################################

# import cv2
# import numpy as np
# import pandas as pd

# # Crear un DataFrame vacío para almacenar las partículas detectadas
# particles = pd.DataFrame(columns=['frame', 'id', 'position'])

# # Crear una lista vacía para almacenar las trazas
# tracks = []

# # Configuración del algoritmo de seguimiento de trazas
# max_distance = 50 # Distancia máxima entre una partícula y una traza para asociarlas
# max_inactive_time = 10 # Número máximo de fotogramas de inactividad antes de terminar una traza

# # Capturar la transmisión de video de la cámara
# cap = cv2.VideoCapture(0)
# # cap = cv2.VideoCapture('cloud.mp4')
# # start_time = 120  # tiempo de inicio en segundos
# # cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

# # Iniciar el ciclo principal
# while True:
#     # Leer un fotograma desde la cámara
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Convertir el fotograma a escala de grises y aplicar un umbral para detectar las partículas
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

#     # Encontrar los contornos de las partículas y agregarlos al DataFrame
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         position = (int(x + w/2), int(y + h/2))
#         particles = particles.append({'frame': cap.get(cv2.CAP_PROP_POS_FRAMES), 'id': len(particles), 'position': position}, ignore_index=True)

#     # Actualizar la lista de trazas
#     for track in tracks:
#         # Buscar la partícula más cercana a la traza
#         min_distance = float('inf')
#         closest_particle = None
#         for _, particle in particles.iterrows():
#             distance = np.sqrt((particle['position'][0] - track['points'][-1][0])**2 + 
#                                (particle['position'][1] - track['points'][-1][1])**2)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_particle = particle
        
#         # Si se encontró una partícula cercana, actualizar la traza
#         if closest_particle is not None and min_distance < max_distance:
#             track['points'].append(closest_particle['position'])
#             track['inactive_time'] = 0
#             particles.drop(closest_particle.name, inplace=True)
#         # Si no se encontró una partícula cercana, aumentar el tiempo de inactividad de la traza
#         else:
#             track['inactive_time'] += 1
        
#         # Si la traza ha estado inactiva durante demasiado tiempo, terminarla
#         if track['inactive_time'] > max_inactive_time:
#             tracks.remove(track)

#     # Crear nuevas trazas a partir de partículas restantes
#     for _, particle in particles.iterrows():
#         tracks.append({
#             'points': [particle['position']],
#             'inactive_time': 0
#         })

#     # Crear una copia del fotograma para dibujar las líneas
#     frame_copy = frame.copy()

#     # Dibujar las líneas de las trazas sobre el fot
#     for track in tracks:
#         for i in range(len(track['points']) - 1):
#             cv2.line(frame_copy, track['points'][i], track['points'][i+1], (0, 255, 0), thickness=2)

#     # Difuminar la imagen de fondo para crear un efecto de estela
#     alpha = 0.3
#     blurred_frame = cv2.blur(frame_copy, (15, 15))
#     blended = cv2.addWeighted(frame, alpha, blurred_frame, 1 - alpha, 0)

#     # Mostrar la imagen resultante en una ventana
#     cv2.imshow('frame', blended)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

######################################################################

# import cv2
# import numpy as np

# # Lista para almacenar las partículas detectadas
# particles = []

# # Configurar la captura de video desde la cámara
# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('cloud.mp4')
# start_time = 120  # tiempo de inicio en segundos
# cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

# # Variables para la detección de bordes y contornos
# reference_frame = None
# reference_edges = None
# reference_contours = None

# # Loop principal
# while True:
#     # Leer un frame desde la cámara
#     ret, frame = cap.read()

#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Si es el primer frame, establecerlo como la imagen de referencia
#     if reference_frame is None:
#         reference_frame = gray
#         reference_edges = cv2.Canny(reference_frame, 50, 150)
#         reference_contours, hierarchy = cv2.findContours(reference_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         continue

#     # Buscar las partículas en el frame actual
#     edges = cv2.Canny(gray, 50, 150)
#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Diccionario para buscar rápidamente partículas existentes por posición
#     existing_particles = {}
#     for particle in particles:
#         existing_particles[(particle[1], particle[2])] = particle[0]

#     # Actualizar las partículas detectadas en la lista
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         if (x, y) in existing_particles:
#             # Actualizar la posición de la partícula existente
#             particle_id = existing_particles[(x, y)]
#             particles[particle_id][1] = x
#             particles[particle_id][2] = y
#         else:
#             # Nueva partícula detectada
#             new_particle = [len(particles), x, y, 0]
#             particles.append(new_particle)

#     # Actualizar el número de frame para cada partícula
#     for particle in particles:
#         particle[3] += 1

#     # Eliminar las partículas que ya no están presentes en la imagen
#     particles = [particle for particle in particles if particle[3] < 50]

#     # Mostrar las partículas en el frame actual
#     frame_copy = frame.copy()
#     for particle in particles:
#         x, y = particle[1], particle[2]
#         cv2.circle(frame_copy, (x, y), 3, (0, 0, 255), -1)
#         if particle[3] > 1:
#             prev_x, prev_y = particle[1], particle[2]
#             for i in range(particle[3] - 1):
#                 prev_x, prev_y = particles[particle[0]][1], particles[particle[0]][2]
#                 cv2.line(frame_copy, (x, y), (prev_x, prev_y), (0, 255, 0), 1)

#     # Mostrar el frame actual
#     cv2.imshow('frame', frame_copy)

#     # Salir si se presiona la tecla 'q'
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Liberar la captura de video y cerrar la ventana de OpenCV
# cap.release()
# cv2.destroyAllWindows()

################################################################################################3

# from pydub import AudioSegment
# from pydub.playback import play

# # Cargar un archivo de sonido
# sound = AudioSegment.from_wav('mi_archivo_de_sonido.wav')

# # Reproducir el sonido
# play(sound)

##################################################################################################

# import cv2
# import numpy as np
# import pandas as pd
# import math

# # Función para detectar trazas
# def detectar_trazas(camara):
#     # Crear un objeto de captura de vídeo
#     cap = cv2.VideoCapture(camara)

#     # Inicializar variables
#     first_frame = None
#     puntos = []
#     trayectorias = []

#     # Configurar las propiedades de la cámara
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     # Iniciar el ciclo de análisis de fotogramas
#     while True:
#         # Obtener el fotograma actual
#         ret, frame = cap.read()

#         # Si no se puede obtener un fotograma, salir del bucle
#         if not ret:
#             break

#         # Convertir el fotograma a escala de grises
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Aplicar un filtro gaussiano al fotograma para reducir el ruido
#         gray = cv2.GaussianBlur(gray, (21, 21), 0)

#         # Si este es el primer fotograma, guardarlo como referencia
#         if first_frame is None:
#             first_frame = gray
#             continue

#         # Calcular la diferencia entre el primer fotograma y el actual
#         frame_delta = cv2.absdiff(first_frame, gray)

#         # Aplicar un umbral a la imagen para detectar cambios significativos
#         thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

#         # Dilatar el resultado del umbral para cubrir agujeros en la traza
#         thresh = cv2.dilate(thresh, None, iterations=2)

#         # Encontrar los contornos de las trazas en el umbral
#         cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Analizar cada contorno
#         for c in cnts:
#             # Si el contorno es demasiado pequeño, ignorarlo
#             if cv2.contourArea(c) < 100:
#                 continue

#             # Calcular los momentos del contorno para obtener la posición
#             M = cv2.moments(c)
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#             puntos.append((cx, cy))

#         if len(puntos) >= 2:
#             # Ajustar una línea a los puntos usando regresión lineal
#             vx, vy, x, y = cv2.fitLine(np.array(puntos), cv2.DIST_L2, 0, 0.01, 0.01)

#             # Calcular el ángulo de la línea
#             angulo = math.atan2(vy, vx) * 180 / math.pi

#             # Agregar la trayectoria al DataFrame
#             trayectorias.append({'x': x, 'y': y, 'angulo': angulo})

#             # Reiniciar la lista de puntos
#             puntos = []

#         # Mostrar los fotogramas en una ventana
#         cv2.imshow('frame', frame)
#         cv2.imshow('thresh', thresh)

#         # Salir del ciclo si se presiona la tecla 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Liberar los recursos utilizados
#     cap.release()
#     cv2.destroyAllWindows()

#     # Convertir las trayectorias a un DataFrame de Pandas y devolverlo
#     return pd.DataFrame(trayectorias)

# df_trayectorias = detectar_trazas(0)

########################################################################################

# import cv2
# import numpy as np

# # Capture video from default camera
# cap = cv2.VideoCapture(0)

# # Define the lower and upper bounds of the line color
# line_color_lower = np.array([0, 0, 0])    # Black color
# line_color_upper = np.array([180, 255, 50])    # Yellowish color

# # Initialize previous centroid point
# prev_centroid = None

# while True:
#     # Read a frame from the video stream
#     ret, frame = cap.read()

#     # Convert the frame to HSV color space
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Threshold the image to get only the line color
#     mask = cv2.inRange(hsv_frame, line_color_lower, line_color_upper)

#     # Find contours in the binary image
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Find the largest contour (should be the line)
#     largest_contour = max(contours, key=cv2.contourArea)

#     # Find the centroid of the largest contour
#     M = cv2.moments(largest_contour)
#     if M["m00"] != 0:
#         centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#     else:
#         centroid = None

#     # Draw the centroid point on the frame
#     if centroid is not None:
#         cv2.circle(frame, centroid, 5, (0, 255, 0), -1)

#         # If there is a previous centroid, check if the current centroid has moved a grossor length away from it
#         if prev_centroid is not None and np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) > grossor_length:
#             # If the current centroid is farther away than the grossor length, the end of the line has been reached
#             print("End of line reached!")

#     # Show the frame
#     cv2.imshow("Frame", frame)

#     # Store the current centroid as the previous centroid for the next iteration
#     prev_centroid = centroid

#     # Check if the user has pressed the 'q' key to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

#############################################################################

# import cv2
# import numpy as np

# # Función para detectar partículas en un frame
# def detect_particles(frame):
#     # Convertir a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Aplicar un filtro gaussiano para reducir el ruido
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Umbralizar la imagen para obtener una máscara binaria
#     _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

#     # Buscar contornos en la máscara binaria
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Crear una lista para almacenar las partículas detectadas
#     particles = []

#     # Iterar sobre todos los contornos encontrados
#     for contour in contours:
#         # Obtener el rectángulo delimitador de cada contorno
#         x, y, w, h = cv2.boundingRect(contour)

#         # Si el rectángulo es suficientemente grande, se considera una partícula
#         if w > 10 and h > 10:
#             particles.append((x + w // 2, y + h // 2))

#     return particles

# # Crear un objeto VideoCapture para leer el vídeo
# cap = cv2.VideoCapture("video.mp4")

# # Crear un objeto para almacenar el camino de las partículas
# path = []

# # Iterar sobre todos los frames del vídeo
# while True:
#     # Leer un frame del vídeo
#     ret, frame = cap.read()

#     # Si no se pudo leer un frame, salir del bucle
#     if not ret:
#         break

#     # Detectar las partículas en el frame actual
#     particles = detect_particles(frame)

#     # Dibujar los centroides de las partículas en el frame
#     for particle in particles:
#         cv2.circle(frame, particle, 5, (0, 0, 255), -1)

#     # Actualizar el camino de las partículas
#     path.append(particles)

#     # Mostrar el frame actual
#     cv2.imshow("Frame", frame)

#     # Esperar 10ms para que se pueda visualizar la imagen
#     if cv2.waitKey(10) == ord("q"):
#         break

# # Convertir el camino a un array de NumPy para poder guardarlo
# path = np.array(path)

# # Calcular el ángulo respecto al eje Y del camino
# angles = np.arctan2(path[:, :, 1], path[:, :, 0])

# # Guardar el camino y el ángulo en archivos separados
# np.save("path.npy", path)
# np.save("angles.npy", angles)

# # Liberar los recursos utilizados
# cap.release()
# cv2.destroyAllWindows()

################################################################################

# import cv2
# import numpy as np

# # Función para detectar partículas en un frame
# def detect_particles(frame):
#     # Convertir a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Aplicar un filtro gaussiano para reducir el ruido
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Umbralizar la imagen para obtener una máscara binaria
#     _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

#     # Buscar contornos en la máscara binaria
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Crear una lista para almacenar las partículas detectadas
#     particles = []

#     # Iterar sobre todos los contornos encontrados
#     for contour in contours:
#         # Obtener el rectángulo delimitador de cada contorno
#         x, y, w, h = cv2.boundingRect(contour)

#         # Si el rectángulo es suficientemente grande, se considera una partícula
#         if w > 10 and h > 10:
#             particles.append((x + w // 2, y + h // 2))

#     return particles

# # Crear un objeto VideoCapture para leer el vídeo
# # cap = cv2.VideoCapture("cloud.mp4")
# cap = cv2.VideoCapture('cloud.mp4')
# start_time = 120  # tiempo de inicio en segundos
# cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)


# # Crear una lista para almacenar el camino de las partículas
# path = []

# # Iterar sobre todos los frames del vídeo
# while True:
#     # Leer un frame del vídeo
#     ret, frame = cap.read()

#     # Si no se pudo leer un frame, salir del bucle
#     if not ret:
#         break

#     # Detectar las partículas en el frame actual
#     particles = detect_particles(frame)

#     # Dibujar los centroides de las partículas en el frame
#     for particle in particles:
#         cv2.circle(frame, particle, 10, (0, 0, 255), -1)

#     # Actualizar el camino de las partículas
#     path.append(particles)

#     # Mostrar el frame actual
#     cv2.imshow("Frame", frame)

#     # Esperar 10ms para que se pueda visualizar la imagen
#     if cv2.waitKey(10) == ord("q"):
#         break

# # Convertir el camino a un array tridimensional de NumPy para poder guardarlo
# # path = np.array(path)

# # Calcular el ángulo respecto al eje Y del camino
# # angles = np.arctan2(path[:, :, :, 1], path[:, :, :, 0])

# # Guardar el camino y el ángulo en archivos separados
# # np.save("path.npy", path)
# # np.save("angles.npy", angles)

# # Liberar los recursos utilizados
# cap.release()
# cv2.destroyAllWindows()