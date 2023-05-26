import os

import numpy as np
import pygame
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as logis

from mnist.mnist import load_dataset

import pickle

pygame.init()

class mnist_visualizer:
    def __init__(self, display):
        self.display = display
        self.exit = False

        self.grid = np.zeros((28, 28))

        self.setup_mnist()

    def update(self):
        self.display.fill((0, 0, 0))

        while not self.exit:
            if pygame.mouse.get_pressed()[0]:
                self.check_mouse_click(mode='draw')
            if pygame.mouse.get_pressed()[2]:
                self.check_mouse_click(mode='erase')

            for event in pygame.event.get():

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.grid = np.zeros((28, 28))
                if event.type == pygame.QUIT:
                    self.exit = True

            self.draw_grid()
            self.draw_info()
            pygame.display.flip()

    def check_mouse_click(self, mode='draw'):
        # if the mouse was clicked or is held down, check if it was inside the grid
        # set the corrosponding pixel to 1 in self.grid
        # if the mouse was clicked outside the grid, return False

        draw = True if mode == 'draw' else False

        # get the mouse position
        mouse_pos = pygame.mouse.get_pos()
        # check if the mouse is inside the grid
        if mouse_pos[0] > 20 and mouse_pos[0] < 580 and mouse_pos[1] > 20 and mouse_pos[1] < 580:

            # calculate the position of the mouse relative to the grid
            relative_pos = (mouse_pos[0] - 20, mouse_pos[1] - 20)
            # calculate the position of the mouse relative to the pixels
            pixel_pos = (int(relative_pos[0] / 20), int(relative_pos[1] / 20))
            # add 1 to the corrosponding pixel
            #self.grid[pixel_pos[0], pixel_pos[1]] += 1 if draw else 0
            # add scaled value to nearby pixels
            add = np.array([[0.2, 0.5, 0.2], [0.5, 1, 0.5], [0.2, 0.5, 0.2]])
            try:
                self.grid[pixel_pos[0] - 1:pixel_pos[0] + 2, pixel_pos[1] - 1:pixel_pos[1] + 2] += add * 0.5 if draw else 0
                # set the max of all pixels to 1
                self.grid[self.grid > 1] = 1
            except:
                pass


    def draw_grid(self):
        # draw a grid of 28x28 pixels, each pixel being a square
        # leave a padding of 50px on the top and bottom
        offset = (20, 20)
        size = (560, 560)
        # draw the grid
        for i in range(28):
            for j in range(28):
                # draw square with color depending on the value of the corrosponding pixel
                color = (self.grid[i, j] * 255,self.grid[i, j] * 255, self.grid[i, j] * 255)

                pygame.draw.rect(self.display, (60, 60, 60), (offset[0] + i * 20, offset[1] + j * 20, 20, 20), 1)
                pygame.draw.rect(self.display, color, (offset[0] + i * 20 + 1, offset[1] + j * 20 + 1, 18, 18), 0)

    def draw_info(self):
        # draw the info text on the right side of the grid
        offset = (600, 20)
        size = (180, 560)
        # draw the info box
        pygame.draw.rect(self.display, (60, 60, 60), (offset[0], offset[1], size[0], size[1]), 0)
        # draw the info text
        words = ['Left click to draw', 'Right click to erase', 'Press space to clear']
        font = pygame.font.SysFont('arial', 20)
        for i in range(len(words)):
            text = font.render(words[i], True, (255, 255, 255))
            self.display.blit(text, (offset[0] + 10, offset[1] + 10 + i * 30))



        most_likely = self.get_info()
        total = np.sum(most_likely[0])
        for i in range(len(most_likely[0])):
            text = font.render(str(i) + ': ' + str(round(most_likely[0][i] / total * 100, 2)) + '%', True, (255, 255, 255))
            self.display.blit(text, (offset[0] + 10, offset[1] + 150 + i * 30))
            pygame.draw.rect(self.display, (100, 100, 100), (offset[0] + 10, offset[1] + 150 + i * 30 + 20, 150, 10), 0, border_radius=10)
            pygame.draw.rect(self.display, (150, 150, 150), (offset[0] + 10, offset[1] + 150 + i * 30 + 20, most_likely[0][i] / total * 150, 10), 0, border_radius=10)



    def setup_mnist(self):
        # load pretrained model from pickle file

        # if file doesnt exist train a new model
        if not os.path.isfile('clf.pkl'):

            text = 'Loading MNIST'
            font = pygame.font.SysFont('arial', 30)
            text = font.render(text, True, (255, 255, 255))
            size = text.get_size()
            screen_size = self.display.get_size()
            self.display.blit(text, (screen_size[0] / 2 - size[0] / 2, screen_size[1] / 2 - size[1] / 2))

            pygame.display.flip()

            # save the model to a pickle file
            X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()

            x_shaped = X_train.reshape(50000, -1)
            y_train = y_train

            self.clf = logis()
            x = self.clf.fit(x_shaped, y_train.T)

            # save self.clf to pickle file
            pickle.dump(self.clf, open('clf.pkl', 'wb'))
        else:
            # load the model from the pickle file
            self.clf = pickle.load(open('clf.pkl', 'rb'))

    def get_info(self):
        input = self.grid
        input = input.reshape(1, -1)
        return self.clf.predict_proba(input)
