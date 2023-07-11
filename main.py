import torch as nn
import cv2
import NeuralNetwork

class AutonomousCar():

    def __init__(self):
    #initialize all variables

    def img_processing(self, image):
        image = cv2.resize(3, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) #rearrange image to (channels, height, width)
        image = nn.from_numpy(image).float() #convert to tensor
        image = self.normalize(image)
        return image

    #converts a value between -1 and 1 to an angle between 0-180
    def steer_angle(self, param):
        y = round((30 - 70) * param + 1 / 1 + 1 + 70, 2)
        return y

    def autopilot(self):
        #some stuff to drive autonomously
        return










