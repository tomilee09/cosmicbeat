import cv2
import numpy as np
import scipy as sp
from contours import join_contours
import MySQLdb as msd
import time as t
start = t.time()
cap = cv2.VideoCapture("cloud.mp4")
cap.open("cloud.mp4")
def grayscale(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img , cv2.COLOR_GRAY2BGR)
    return img
def getLineFromContour(contour , mask):
    center , sides , angle = cv2.minAreaRect(contour)
    x, y = center
    l, m = sides
    theta = np.radians(angle)
    if l > m:
        a =(int(x - l*np.cos(theta)), int(y - m*np.sin(theta)))
        b =(int(x + l*np.cos(theta)), int(y + m*np.sin(theta)))
    else :
        a =(int(x - m*np.sin(theta)), int(y + m*np.cos(theta)))
        b =(int(x + m*np.sin(theta)), int(y - m*np.cos(theta)))
    length = np.linalg.norm(np.array(a)-np.array(b))
    matrix = cv2.getRotationMatrix2D(center , angle , 1)
    rotated = cv2.warpAffine(mask , matrix , mask.shape [:2])
    rect = cv2.getRectSubPix(rotated ,(int(l), int(m)), center)
    intensity = 0
    if l*m != 0:
        intensity = float(cv2.countNonZero(rect))/(l*m)
        divergence = 0
    if length != 0:
        divergence =(l*m)/(length * length)
        return a, b, angle , intensity , length , divergence
def detectTracks(img):
    mask = fgbg.apply(img)
    mask = cv2.blur(mask ,(15 , 15))
    ret , mask = cv2.threshold(mask , 64 , 255 , cv2.THRESH_BINARY)
    mask = cv2.dilate(mask ,(4 ,4))
    maskstash = mask.copy()
    i, contours , hierarchy = cv2.findContours(mask , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours)> 1:
        contours = join_contours(contours)
    return maskstash , [ getLineFromContour(c, maskstash)for c in contours ]
def linesSimilar(la , lb):
    ((a1x , a1y),(a2x , a2y), angleA , intensity , lengthA , divergence)= la
    ((b1x , b1y),(b2x , b2y), angleB , intensity , lengthB , divergence)= lb
    anglediff = abs(angleA - angleB)
    if anglediff > 30:
        return False
    a1 = np.array(la[0])
    a2 = np.array(la[1])
    b1 = np.array(lb[0])
    b2 = np.array(lb[1])
    dist1 = np.linalg.norm(a1 - b1)> 50
    dist2 = np.linalg.norm(a2 - b2)> 50
    dist3 = np.linalg.norm(a1 - b2)> 50
    dist4 = np.linalg.norm(a2 - b1)> 50
    if sum((dist1 , dist2 , dist3 , dist4))< 2:
        return False
    return True
def analyzeEvents(eventDict):
    rows = []
    for k, value in eventDict.items():
        event_id = k
        timestamp = value [0][0]
        age = value [ -1][0] - value [0][0]
        lines = [l for t,l in value ]
        angle = lines [3][2]
        lengths = np.array([l[4] for l in lines ])
        length = np.percentile(lengths , 90)
        intensities = np.array([l[3] for l in lines ])
        intensity = np.percentile(intensities , 90)
        divergences = np.array([l[5] for l in lines ])
        divergence = np.percentile(divergences , 20)
        rows += [(event_id , angle , length , intensity ,
        timestamp , "?", age , divergence)]
    return rows
def enterEvents(events):
    rows = analyzeEvents(events)
    connection = msd.connect(" localhost ", " user ", " password ", " database ")
    with connection:
        cursor = connection.cursor()
        for r in rows:
            cursor.execute(" insert into events values "+ str(r)+";")
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg.setVarThreshold(30)
time = 0
nextid = 0
eventlist = {}
events = {}
frame = None
gray = None
mask = None
while(cap.isOpened()):
    try :
        ret , frame = cap.read()
        if not ret :
            break
        gray = grayscale(frame)
        mask , lines = detectTracks(gray)
        currentevents = []
        if time != 0:
            for l in lines :
                found = False
                for l2 in eventlist [time -1]:
                    if linesSimilar(l, l2 [1]):
                        currentevents += [(l2 [0] , l)]
                        events [l2 [0]] += [(time , l)]
                        found = True
                        break
            if not found :
                currentevents += [(nextid , l)]
                events [ nextid ] = [(time ,l)]
                nextid += 1
        else :
            for l in lines :
                currentevents += [(nextid , l)]
                events [ nextid ] = [(time , l)]
                nextid += 1
        eventlist [ time ] = currentevents
    except Exception as e:
        print(time , e)
    finally :
        time += 1
events = {k: v for k,v in events.items()if len(v)> 4}
enterEvents(events)
cap.release()
cv2.destroyAllWindows()
end = t.time()
print(time /(end - start)," frames /second ,")
print(len(events.values())," events processed in" ,(end - start)," seconds ")