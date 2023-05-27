import time
import matplotlib.pyplot as plt
import numpy as np

import configparser

import pygame

config = configparser.ConfigParser()
config.read('config.ini')


class gradient_descent_visualizer:
    def __init__(self, display):
        self.exit = False
        self.display = display
        self.order = 2

        self.reset()

    def generate_data(self):
        zero_min = config.getfloat('gradient_descent', 'zeros_min')
        zero_max = config.getfloat('gradient_descent', 'zeros_max')

        zeros = np.zeros((self.order))
        for i in range(self.order):
            zeros[i] = np.random.randint(zero_min * 10, zero_max * 10) / 10

        zeros = np.array([4, 4])

        self.data = np.zeros((100, 2))
        for i in range(100):
            x = i / 10 - 10
            y = 0.05
            for j in range(self.order):
                y *= x - zeros[j]
            # add noise
            # self.data[i] = [x + np.random.normal(
            #    0, 0.1), y + np.random.normal(0, 0.1)]
            self.data[i] = [x, y]

    def reset(self):
        self.generate_data()
        self.phi = 0.5
        self.theta = 0.33

        zero_min = config.getfloat('gradient_descent', 'zeros_min')
        zero_max = config.getfloat('gradient_descent', 'zeros_max')

        self.ball_pos = np.array([np.random.randint(zero_min * 10, zero_max * 10) / 10,
                                  np.random.randint(zero_min * 10, zero_max * 10) / 10])

        self.populate_gradients()

    def update(self):
        while not self.exit:
            if pygame.mouse.get_pressed()[0]:
                rel = pygame.mouse.get_rel()

                if rel[1] < 10 or rel[0] < 10:
                    # self.theta += rel[1] / 100
                    self.phi += rel[0] / 100

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.reset()

                if event.type == pygame.QUIT:
                    self.exit = True

            self.draw()

    def draw(self):
        self.draw_data()
        self.draw_descent()
        self.draw_info()
        pygame.display.flip()

    def draw_estimated(self, surface):
        points = []
        for i in range(100):
            x = i / 10 - 5
            y = 0.05
            for j in range(self.order):
                y *= x - self.ball_pos[j]

            points.append((300 + x * 60, int(-y * 100 + 67.5)))

        pygame.draw.lines(surface, (255, 0, 0), False, points, 1)

    def draw_data(self):
        self.display.fill((0, 0, 0))
        surface = pygame.Surface((600, 125))
        pygame.draw.rect(surface, (50, 50, 50), (0, 0, 600, 125), border_radius=10)
        for i in range(self.data.shape[0]):
            x = self.data[i][0]
            y = self.data[i][1]
            pygame.draw.circle(surface, (255, 255, 255), (300 + x * 60, int(-y * 100 + 67.5)), 2)

        self.draw_estimated(surface)

        self.display.blit(surface, (25, 450))

    def draw_descent_grid(self, surface):

        grid_color = (100, 100, 100)

        for r in range(10, 250, 30):
            pygame.draw.ellipse(surface, grid_color, (300 - r, 300 - r * self.theta, r * 2, r * 2 * self.theta), 1)

        for i in range(0, 360, 360 // 8):
            angle = i / 180 * np.pi + self.phi
            pygame.draw.line(surface, grid_color, (300, 300),
                             (300 + 220 * np.cos(angle), 300 + 220 * np.sin(angle) * self.theta), 1)

    def draw_descent(self):

        surf = pygame.Surface((600, 400), pygame.SRCALPHA)
        pygame.draw.rect(surf, (20, 20, 20), (0, 0, 600, 400), border_radius=10)
        self.draw_descent_grid(surf)

        def get_point(r, theta):
            x = r * np.cos(theta / 36 * np.pi * 2) * 10
            y = r * np.sin(theta / 36 * np.pi * 2) * 10
            z = -self.gradients[theta % 36][r]

            return np.array([x, y, z])

        def get_color(p1, p2, p3):
            grad = np.linalg.norm(p1 - p2) + np.linalg.norm(p2 - p3) + np.linalg.norm(p3 - p1)
            grad *= 2
            grad = np.clip(grad, 0, 255)
            if grad == np.nan:
                grad = 0


            return grad, 255 - grad, 25

        for r in range(20):
            # only draw half of the points
            # do the half that is on the opposite side of the sphere
            # this means self.phi + 90 degrees to self.phi - 90 degrees
            #for theta in range(int(-self.phi / (np.pi * 2) * 36 + 18), int((-self.phi + np.pi) / (np.pi * 2) * 36) + 24):
            for theta in range(36):
                p = get_point(r, theta)
                p_above = None
                p_below = None
                p_right = None
                p_left = None
                if r < 19:
                    p_above = get_point(r + 1, theta)

                p_right = get_point(r, theta + 1)

                if r > 0:
                    p_below = get_point(r - 1, theta)

                p_left = get_point(r, theta - 1)

                if (r % 2 == 0 and theta % 2 == 0) or (r % 2 == 1 and theta % 2 == 1):
                    if p_above is not None and p_right is not None:
                        points = [self.translate_in_3d(p), self.translate_in_3d(p_above), self.translate_in_3d(p_right)]
                        pygame.draw.polygon(surf, get_color(p, p_above, p_right), points)

                    if p_below is not None and p_right is not None:
                        points = [self.translate_in_3d(p), self.translate_in_3d(p_below), self.translate_in_3d(p_right)]
                        pygame.draw.polygon(surf, get_color(p, p_below, p_right), points)

                    if p_below is not None and p_left is not None:
                        points = [self.translate_in_3d(p), self.translate_in_3d(p_below), self.translate_in_3d(p_left)]
                        pygame.draw.polygon(surf, get_color(p, p_below, p_left), points)

                    if p_above is not None and p_left is not None:
                        points = [self.translate_in_3d(p), self.translate_in_3d(p_above), self.translate_in_3d(p_left)]
                        pygame.draw.polygon(surf, get_color(p, p_above, p_left), points)

        ball_pos = self.ball_pos
        r = round(np.sqrt(ball_pos[0] ** 2 + ball_pos[1] ** 2))
        theta = round(np.arctan2(ball_pos[1], ball_pos[0]) / np.pi *18) % 36
        p = np.array([ball_pos[0] * 10, ball_pos[1] * 10, -self.gradients[theta][r]])
        p = self.translate_in_3d(p)
        pygame.draw.circle(surf, (255, 255, 255), (int(p[0]), int(p[1])), 5)

        minpos = self.minpos
        r = round(np.sqrt(minpos[0] ** 2 + minpos[1] ** 2))
        theta = round(np.arctan2(minpos[1], minpos[0]) / np.pi * 18) % 36
        p = np.array([minpos[0] * 10, minpos[1] * 10, -self.gradients[theta][r]])
        p = self.translate_in_3d(p)
        pygame.draw.circle(surf, (100, 255, 0), (int(p[0]), int(p[1])), 5, 2)

        self.move_ball()

        self.display.blit(surf, (25, 25))

    def move_ball(self):
        ball_pos = self.ball_pos
        r = round(np.sqrt(ball_pos[0] ** 2 + ball_pos[1] ** 2))
        theta = round(np.arctan2(ball_pos[1], ball_pos[0]) / np.pi * 18) % 36
        # move the ball in the direction of the lowest height, with scaled speed
        target = np.array([0.0, 0.0])
        for r_add in range(-1, 2):
            for theta_add in range(-1, 2):
                if r + r_add >= 0 and r + r_add < 20 and theta + theta_add >= 0 and theta + theta_add < 36:
                    x = (r + r_add) * np.cos((theta + theta_add) / 18 * np.pi)
                    y = (r + r_add) * np.sin((theta + theta_add) / 18 * np.pi)
                    if self.get_height(x, y) < self.gradients[theta][r]:
                        target = np.array([x - ball_pos[0], y - ball_pos[1]])

        self.ball_pos += target * 0.1





    def draw_info(self):
        text = [
            'Ball x: ' + str(round(self.ball_pos[0], 2)),
            'Ball y: ' + str(round(self.ball_pos[1], 2)),
            f'x^2 + {round(self.ball_pos[0] + self.ball_pos[1], 2)}x + {round(self.ball_pos[0] * self.ball_pos[1], 2)}',
            "Score: " + str(self.get_height(self.ball_pos[0], self.ball_pos[1])),
        ]
        self.font = pygame.font.SysFont('Arial', 20)
        for i in range(len(text)):
            self.display.blit(self.font.render(text[i], True, (255, 255, 255)), (650, 25 + i * 25))

    def populate_gradients(self):
        self.gradients = np.zeros((36, 20))
        minx = 1000
        miny = 1000
        minval = 1000000000

        for theta in range(36):
            for r in range(20):
                x = r * np.cos(theta / 36 * np.pi * 2) / 2
                y = r * np.sin(theta / 36 * np.pi * 2) / 2
                z = self.get_height(x, y)

                if z < minval:
                    minval = z
                    minx = x
                    miny = y

                self.gradients[theta][r] = z

        self.minpos = np.array([minx, miny])




        # get max height
        max_height = np.max(self.gradients)
        # normalize
        self.gradients /= max_height
        self.gradients = self.gradients * 500

    def get_height(self, x1, x2):
        h = 0
        point = np.array([x1, x2])
        for i in range(self.data.shape[0]):
            y = 0.05 * (self.data[i][0] - point[0]) * (self.data[i][0] - point[1])  # get estimated height at data point
            deltay = self.data[i][1] - y  # get difference between estimated height and actual height

            h += deltay * deltay
        return h

    def translate_in_3d(self, point):
        # rotate a given point given a phi and theta
        # phi is the angle in the x-y plane
        # theta is the angle in the x-z plane

        # m is the 3d translation matrix with phi and theta
        x = point[0]
        y = point[1]
        z = point[2]

        xnew = (x * np.cos(self.phi) - y * np.sin(self.phi)) * np.cos(self.theta)
        ynew = (x * np.sin(self.phi) + y * np.cos(self.phi)) * np.cos(self.theta)
        znew = z * np.sin(self.theta) + ynew * np.sin(self.theta)

        x = xnew
        y = ynew
        z = znew

        return np.array([x + 300, z + 300])
