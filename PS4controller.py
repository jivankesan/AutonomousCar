import os
import pygame

'''
Class for ps4 controller to work with Jetson Nano
'''

class controller():

    def __init__(self):
        '''
        initializing variables
        '''
        pygame.init()
        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
