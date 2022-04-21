import cv2 as cv

imge = cv.imread("img/fampic.jpg")

def resizedimge(obj):
    w = int(obj.shape[1]*0.75)
    h = int(obj.shape[0]*0.75)
    return cv.resize(obj, (w, h), interpolation=cv.INTER_AREA)

imgeResized = resizedimge(imge)
imgeGray = cv.cvtColor(imgeResized, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier("haarcascade_frontalface.xml")

face_detect = haar_cascade.detectMultiScale(imgeGray, scaleFactor=1.1, minNeighbors=6)

print('Number of faces detected: ', len(face_detect))
print(face_detect)

for(x1, y1, x2, y2) in face_detect:
    cv.rectangle(imgeResized, (x1, y1), (x1+x2, y1+y2), (0,255,0), thickness=1)

cv.imshow("Photo", imgeResized)
cv.waitKey(0) 