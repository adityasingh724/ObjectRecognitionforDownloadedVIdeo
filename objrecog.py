
#Make sure to save the downloaded video in the same folder as the project


import cv2

# Object Detection Software

print("Developer: Aditya Singh")


thres = 0.5 #Threshold to detect objects


cap = cv2.VideoCapture(" ") #Enter the filename for the video here
classNames = []
classFile = 'Labels.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)

net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)


    print(classIds,bbox)

    if len(classIds) != 0:

        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color = (255,0,0), thickness = 2)
            cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            



    cv2.imshow("Object Detection Software", img)
    cv2.waitKey(1)
