# OpenCV Python program to detect cars in video frame 
# import libraries of python OpenCV  
import cv2 
import numpy as np

# capture frames from a video 
cap = cv2.VideoCapture('new.mp4') 

fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(cap.get(3)) # Get the Default resolutions
frame_height = int(cap.get(4))

# Trained XML classifiers describes some features of some object we want to detect 
car_cascade = cv2.CascadeClassifier('cars.xml')
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
'''
human1_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
human2_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
'''
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

# loop runs if capturing has been initialized. 
while True: 
    # reads frames from a video 
    ret, frames = cap.read()
    
    if ret == True:
        
        # convert to gray scale of each frames 
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        
        # Detects cars of different sizes in the input image 
        cars = car_cascade.detectMultiScale(gray, 1.1, 5)
        humans = human_cascade.detectMultiScale(gray, 1.1, 4)
        '''
        humans1 = human1_cascade.detectMultiScale(gray, 1.1, 4)
        humans2 = human2_cascade.detectMultiScale(gray, 1.2, 4)
        '''
        
        # To draw a rectangle in each cars 
        for (x,y,w,h) in cars:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(255,0, 0),2) 
        for (x,y,w,h) in humans:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
        '''
        for (x,y,w,h) in humans1:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2) 
        for (x,y,w,h) in humans2:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2) 
        '''
        out.write(frames)
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows() 
