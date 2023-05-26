import time as tme

import pygame

import numpy as np
from sklearn.decomposition import PCA

class dimension_reduction:
    def __init__(self, display):
        self.exit = False
        self.display = display
        self.reset()

        self.theta = 0.5
        self.phi = 0.5
        self.step = 0

    def reset(self):
        num_clusters = 3
        num_points = 100
        # generate gaussian data with num_clusters
        self.data = np.zeros((num_clusters * num_points, 3))
        cov = np.array([[0.2, 0, 0], [0, .1, 0], [0, 0, .001]])
        for i in range(num_clusters):
            self.data[i * num_points:(i + 1) * num_points] = np.random.multivariate_normal(
                np.random.randint(-5, 5, 3), cov, num_points)

        # set all data's 3rd dimension to random between -0.1 and 0.1

        self.data = self.data * 25
        self.data[:, 2] = np.random.uniform(-0.1, 0.1, num_clusters * num_points)

        self.time = 0
        self.step = 0

        self.usvt = np.linalg.svd(self.data)

    def update(self):
        while not self.exit:
            if pygame.mouse.get_pressed()[0]:
                rel = pygame.mouse.get_rel()

                if rel[1] < 10 or  rel[0] < 10:
                    self.theta -= rel[1] / 100
                    self.phi += rel[0] / 100

                    self.theta = max(min(self.theta, np.pi), 0)

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:
                        self.step += 1
                        self.lerp_all()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.reset()

                if event.type == pygame.QUIT:
                    self.exit = True

            self.display.fill((0, 0, 0))

            self.draw_data(data=self.data)
            pygame.display.flip()

    def draw_data(self, data=None):
        surface = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(surface, (255, 255, 255, 120), (5, 5), 5)

        for i, point in enumerate(data):
            p = self.translate_in_3d(point)
            self.display.blit(surface, (p[0]-5, p[1]-5))

        coords = np.array([
            [-100, 0, 0], [100, 0, 0],
            [0, -100, 0], [0, 100, 0],
            [0, 0, -100], [0, 0, 100]
        ])
        for i, point in enumerate(coords):
            colors = [(255, 100, 50), (50, 255, 100), (100, 50, 255)]
            if i % 2 == 0:
                pygame.draw.line(self.display, colors[i // 2], self.translate_in_3d(point), self.translate_in_3d(coords[i + 1]), 2)

    def lerp_all(self):
        step = self.step

        pca = PCA(n_components=3-step)
        pca.fit(self.data)
        tmp = pca.transform(self.data)
        cov = pca.get_covariance()
        print(cov)
        index = np.argmax(cov.diagonal())
        # pad the index'th column with zeros
        #tmp = self.data.copy()
        target = np.zeros((self.data.shape[0], 3))
        if index == 0:
            target[:, 1:] = tmp[:, :]
        elif index == 1:
            target[:, 0] = tmp[:, 0]
            target[:, 2] = tmp[:, 1]
        else:
            target[:, :2] = tmp[:, :]



        tmpdata = self.data.copy()
        while self.time < 1:
            for i in range(self.data.shape[0]):
                tmpdata[i] = lerp(self.data[i], target[i], self.time)
            self.time += 0.01
            tme.sleep(0.01)
            self.display.fill((0, 0, 0))
            self.draw_data(data=tmpdata)
            pygame.display.flip()

        self.data = tmpdata.copy()



    def translate_in_3d(self, point):
        # rotate a given point given a phi and theta
        # phi is the angle in the x-y plane
        # theta is the angle in the x-z plane


        # m is the 3d translation matrix with phi and theta
        m = np.array([[np.cos(self.phi), np.sin(self.phi) * np.sin(self.theta), np.sin(self.phi) * np.cos(self.theta)],
                        [0, np.cos(self.theta), -np.sin(self.theta)],
                        [-np.sin(self.phi), np.cos(self.phi) * np.sin(self.theta), np.cos(self.phi) * np.cos(self.theta)]])
        ans = np.dot(m, point)
        # add 400 and 300 to center the point
        return ans[0] + 400, ans[1] + 300
def lerp(a, b, t):
    return a + (b - a) * 1 / (1 + np.exp(-10*t + 5))