import numpy as np
import imutils
import time
import cv2
# assigned the pretrained model ,text 
prototxt="MobileNetSSD_deploy.prototxt.txt"
model="MobileNetSSD_deploy.caffemodel"
confThresh=0.2# confidence level/Threshold
# accesing / sending the type of objects to be detected by the model in the process
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
# randomingly chossing colors based on length of classes ie during detection the boundary box colors vary for each object in class
COLORS=np.random.uniform(0,255,size=(len(CLASSES), 3))
print("............Loading Model.......")
# reading or deploying the model using the given parameters
net=cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
print("Starting Camera")
vs=cv2.VideoCapture(0)
time.sleep(2.0)# 2 seconds

while True:
    _,frame=vs.read()
    frame=imutils.resize(frame,width=500)# resizing the image using imutils
    (h,w)=frame.shape[:2]# getting the width and height of the image captured
    # creating a Blob based on which Dnn can operate
    imResizeBlob=cv2.resize(frame,(300,300))#resizing the boundary of image as mobile net ssd need 300*300 frame
    blob=cv2.dnn.blobFromImage(imResizeBlob,0.007843,(300,300),127.5)#convert image to blob
    net.setInput(blob)#passing blob image to the model
    detections=net.forward()# training the model in such a way that the image is procesed in the forward direction using accuracy and so on
    detShape=detections.shape[2]# getting shape
    for i in np.arange(0,detShape):
        confidence=detections[0,0,i,2]# represents multiple object detection
        if confidence > confThresh:
            idx=int(detections[0,0,i,1])# returns the class number for classification among thwe above classes so that result text can be displayed
            #plotting box for the object based on the start and stop coordinates
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY)= box.astype("int")# converting points  to int
            label="{}:{:.2f}%".format(CLASSES[idx],confidence*100)# prints name of the class like aeroplane ,and also displays the accuracy like how sure the model knows that object is an aeroplane
            #drawing/making a boundary box for the same
            cv2.rectangle(frame,(startX, startY), (endX, endY),COLORS[idx],2)
            #used for plotting text above the boundary box
            if startY - 15 > 15:
                y = startY-15
            else:
                startY+15
            cv2.putText(frame,label,(startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,COLORS[idx],2)
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key == 27:
        break
vs.release()
cv2.destroyAllWindows()
