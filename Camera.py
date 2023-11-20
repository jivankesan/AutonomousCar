# Import necessary modules: atexit, cv2, threading, and numpy.
import cv2
import os
import numpy as np
import atexit
import threading


# Define the Camera class
# Initialize the class:

class Camera():
    def __init__(self):
        self.image_width = 640
        self.image_height = 640
        self.image_channels = 3
        self.output = np.empty((224, 224, 3), dtype=np.uint8)

        try:
            self.cap = cv2.VideoCapture(0)

            re, img = self.cap.read()

            if not re:
                print("error reading from camera")
                return

            self.output = img
            self.start()

        except:
            self.stop()
            print("issue initializing camera")

        atexit.register(self.stop())

    # Set up a method to capture frames from video feed
    def live_stream(self):
        while True:
            # read current frame from cam
            ret, frame = self.cap.read()
            # check if frame was read successfully
            if not ret:
                break
            else:
                self.output = frame

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
            print('camera function stopped')
        threading.Thread.join()

    def start(self):
        if not self.cap.isOpened():
            self.cap.open()

    def live_training(self):
        video_capture = cv2.VideoCapture(0)
        frame_count = 0

        # Directory path to save each frame
        save_dir = 'path/to/save/directory/'

        # Create the save directory if it doesn't exist on the laptop
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        while True:
            # read current frame from cam
            ret, frame = video_capture.read()

            # check if frame was read successfully
            if not ret:
                break

            frame_count += 1

            # display captured frame
            cv2.imshow('Live Car Feed', frame)

            # Save each frame as an image
            frame_name = f"frame_{frame_count}.jpg"
            save_path = os.path.join(save_dir, frame_name)
            cv2.imwrite(save_path, frame)

            height, width, channels = frame.shape
            captured_image = np.zeros((height, width, channels), dtype=np.uint8)

            # press q to close live feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Total frames captured:", frame_count)

# Start camera capture (read about threading in operating systems)


# figure out how to capture the frames continuously
# make sure you keep track of the count of frames

# create a function to stream live feed to my computer
