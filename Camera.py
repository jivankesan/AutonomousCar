#Import necessary modules: atexit, cv2, threading, and numpy.
import cv2
import os
import numpy as np
import atexit
import threading


#Define the Camera class
#Initialize the class:

class Camera():
 #Set initial values for the camera parameters, such as capture width, capture height, frames per second (fps), and image dimensions.
   def capture_image(image_width, image_height, image_channels):
    #Create an array to store the captured image.
    self.captured_image = np.zeros((image_height, image_width, image_channels), dtype=np.uint8)

#Set up a method to capture frames from video feed
  def live_stream():
    self.video_capture = cv2.VideoCapture(0)

    while True:
      #read current frame from cam
      ret, frame = video_capture.read()

      #check if frame was read successfully
      if not ret:
        break

      #display captured frame
      cv2.imshow('Live Car Feed', frame)

      #press q to close live feed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video.release()
    cv2.destroyAllWindows()




  def live_training():
    video_capture = cv2.VideoCapture(0)
    frame_count = 0

    #Directory path to save each frame
    save_dir = 'path/to/save/directory/'

    #Create the save directory if it doesn't exist on the laptop
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    while True:
      #read current frame from cam
      ret, frame = video_capture.read()

      #check if frame was read successfully
      if not ret:
        break

      frame_count += 1

      #display captured frame
      cv2.imshow('Live Car Feed', frame)

      #Save each frame as an image
      frame_name = f"frame_{frame_count}.jpg"
      save_path = os.path.join(save_dir, frame_name)
      cv2.imwrite(save_path, frame)

      height, width, channels = frame.shape
      captured_image = np.zeros((height, width, channels), dtype=np.uint8)

      #press q to close live feed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video.release()
    cv2.destroyAllWindows()
    print("Total frames captured:", frame_count)



#Start camera capture (read about threading in operating systems)


#figure out how to capture the frames continuously
#make sure you keep track of the count of frames

#create a function to stream live feed to my computer
