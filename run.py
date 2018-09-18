
import cv2
import sys
from em_model import EMR
import numpy as np
import matplotlib.pyplot as plt
import six
import tflearn.helpers.summarizer as s
import serial
import struct
import time
import numpy

SerialData = serial.Serial('/dev/ttyUSB1', 115200)
time.sleep(.1)

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

cascade_classifier = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')

def brighten(data,b):
     datab = data * b
     return datab    

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
  faces = cascade_classifier.detectMultiScale(
      image,
      scaleFactor = 1.3 ,
      minNeighbors = 5
  )
  
  if not len(faces) > 0:
    return None

  max_area_face = faces[0]
  
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
      
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

  try:
    image = cv2.resize(image, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
  return image

network = EMR()
network.build_network()

video_capture = cv2.VideoCapture(0)     #video source!!!
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    
  ret, frame = video_capture.read()
  result = network.predict(format_image(frame))
  
  #print(result)
  
  neuron = int(result[0][3] * 250)

  if neuron <= 9:
    neuron = "00" + str(neuron)

  if neuron >= 10 and neuron <= 99:
    neuron = "0" + str(neuron)

  print(neuron)
  
  #result_layer2 = network.predict2(format_image(frame))
  #result_layer2.sort()
  
  result_layer1 = network.predict3(format_image(frame))
  result_layer1.sort()
  
  #print(result_layer2)
  #print(result_layer1[0][2] * 128)
  
  #neuron0 = int(result_layer1[0][2] * 128)
  #neuron0 = result_layer1[0][2] * 128

  #neuron2 = int(result_layer2[0][30] * 128)
  #new_neuron = int(neuron)
  #print(neuron0)
  #print(result[0][299])
  #print(result[0][0])
  #print(result)
  
  SerialData.write(str.encode(str(neuron)))

  #.write(struct.pack('>B',  neuron))

  if result is not None:
    for index, emotion in enumerate(EMOTIONS):
      cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2);
      cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 150), (index + 1) * 20 + 4), (255, 0, 0), -1)

  cv2.imshow('Video', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()














