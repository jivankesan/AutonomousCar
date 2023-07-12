#call these functions to make "turn" degree turns with servo motor using jetson nano through the pca9685

from adafruit_servokit import ServoKit
myKit = ServoKit(channels=16) #loading up servo with 16 channels
import time

def left_turn_servo(turn):
    for i in range(0,turn,1): #turn left slowly
        myKit.servo[0].angle = i
        time.sleep(0.01)

def right_turn_servo(turn):
    for i in range(turn,0,-1): #turn right slowly
        myKit.servo[0].angle = i
        time.sleep(0.01)

