import time

import pygame
import numpy as np

class kmeans_3D:
    def __init__(self, display):
        self.exit = False
        self.display = display
        self.cluster_count = 5
        self.reset()

        self.theta = 0.5
        self.phi = 0.5

    def reset(self):
        self.last_step = ''
        self.setup_kmeans()
        # set the means to 3 random points in the data
        self.means = self.data[np.random.randint(0, self.data.shape[0], self.cluster_count), :]

    def update(self):
        while not self.exit:
            if pygame.mouse.get_pressed()[0]:
                rel = pygame.mouse.get_rel()

                if rel[1] < 10 or  rel[0] < 10:
                    self.theta += rel[1] / 100
                    self.phi += rel[0] / 100




            for event in pygame.event.get():



                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:
                        self.iterate()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.reset()


                if event.type == pygame.QUIT:
                    self.exit = True

            self.draw()

    def draw(self):
        self.display.fill((0, 0, 0))
        self.draw_data()
        self.draw_means()
        self.show_info()
        pygame.display.flip()

    def draw_data(self):
        colors = [(255, 100, 50, 120), (50, 255, 100, 120), (100, 50, 255, 120), (255, 255, 100, 120), (100, 255, 255, 120), (255, 100, 255, 120)]
        null_circle = pygame.Surface((10, 10), pygame.SRCALPHA)
        data_circle = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(null_circle, (120, 120, 120, 60), (5, 5), 5)
        tmpdata = self.data.copy()
        for i in range(self.data.shape[0]):
            tmpdata[i] = self.translate_in_3d(self.data[i])
        for i in range(self.data.shape[0]):

            if self.assignment[i, 0] == -1:
                self.display.blit(null_circle, (int(tmpdata[i, 0] * 100) + 400 - 5,
                                                 int(tmpdata[i, 1] * 100) + 300 - 5))

            else:
                pygame.draw.circle(data_circle, colors[int(self.assignment[i, 0])], (5, 5), 5)
                self.display.blit(data_circle, (int(tmpdata[i, 0] * 100) + 400 - 5, int(tmpdata[i, 1] * 100) + 300 - 5))

    def draw_means(self):
        tmpmeans = self.means.copy()
        for i in range(self.means.shape[0]):
            tmpmeans[i] = self.translate_in_3d(self.means[i])
        #draw a little cross at the position of the means
        colors = [(255, 25, 12), (12, 255, 25), (25, 12, 255), (255, 255, 25), (25, 255, 255), (255, 25, 255)]
        for i in range(self.means.shape[0]):
            pygame.draw.line(self.display, colors[i], (int(tmpmeans[i, 0] * 100) + 400 - 5,
                                                       int(tmpmeans[i, 1] * 100) + 300 - 5),
                             (int(tmpmeans[i, 0] * 100) + 400 + 5,
                              int(tmpmeans[i, 1] * 100) + 300 + 5), 2)
            pygame.draw.line(self.display, colors[i], (int(tmpmeans[i, 0] * 100) + 400 - 5,
                                                       int(tmpmeans[i, 1] * 100) + 300 + 5),
                             (int(tmpmeans[i, 0] * 100) + 400 + 5,
                              int(tmpmeans[i, 1] * 100) + 300 - 5), 2)

    def setup_kmeans(self):
        self.generate_data()

    def generate_data(self):
        # generate random 2d gaussian data
        # generate 50 points for 3 clusters
        self.data = np.zeros((50 * self.cluster_count, 3))
        self.assignment = np.zeros((50 * self.cluster_count, 1))
        self.assignment[:, 0] = -1
        self.correct_assignment = np.zeros((50 * self.cluster_count, 1))

        # set the means to 3 equidistant points on a circle
        # data clusters must have the same random cov, cov must be smaller than 0.5

        cov = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self.means = np.zeros((self.cluster_count, 3))
        for i in range(self.cluster_count):
            self.means[i, :] = np.array([1.5 * np.cos(2 * np.pi * i / self.cluster_count),
                                         1.5 * np.sin(2 * np.pi * i / self.cluster_count),
                                         0])

        for i in range(self.cluster_count):
            self.data[i * 50:(i + 1) * 50, :] = np.random.multivariate_normal(self.means[i, :], cov, 50)

        for i in range(self.cluster_count):
            self.correct_assignment[i * 50:(i + 1) * 50, :] = i

        #self.iterate()

    def iterate(self):
        if self.last_step != 'estimation':
            self.last_step = 'estimation'
            for j, data_point in enumerate(self.data):
                # calculate the distance to each mean
                distances = np.zeros(self.cluster_count)
                for i in range(self.cluster_count):
                    distances[i] = np.linalg.norm(data_point - self.means[i, :])
                # assign the data point to the index of the closest mean
                if (self.assignment[j] != np.argmin(distances)):

                    self.assignment[j] = np.argmin(distances)
                    self.draw()
                    time.sleep(0.001)
        else:
            self.last_step = 'maximization'
            # calculate the new means
            for i in range(self.cluster_count):
                self.means[i, :] = np.mean(self.data[self.assignment[:, 0] == i, :], axis=0)
            self.draw()

    def show_info(self):
        text = [
            'Space for new set',
            'Drag to move',
            'Right click for next step',
            'Current step: ' + self.last_step
        ]
        self.font = pygame.font.SysFont('arial', 20)
        for i, line in enumerate(text):
            self.display.blit(self.font.render(line, True, (255, 255, 255)), (25, 25 + i * 20))

        colors = [(255, 100, 50, 120), (50, 255, 100, 120), (100, 50, 255, 120), (255, 255, 50, 120), (50, 255, 255, 120), (255, 50, 255, 120)]

        for j, mean in enumerate(self.means):
            # let tmp_mean be the mean but rounded to 2 decimals
            tmp_mean = np.round(mean, 2)
            self.display.blit(self.font.render(str(tmp_mean), True, colors[j]), (25, 110 + j * 20))


    def translate_in_3d(self, point):
        # rotate a given point given a phi and theta
        # phi is the angle in the x-y plane
        # theta is the angle in the x-z plane


        # m is the 3d translation matrix with phi and theta
        m = np.array([[np.cos(self.phi), np.sin(self.phi) * np.sin(self.theta), np.sin(self.phi) * np.cos(self.theta)],
                        [0, np.cos(self.theta), -np.sin(self.theta)],
                        [-np.sin(self.phi), np.cos(self.phi) * np.sin(self.theta), np.cos(self.phi) * np.cos(self.theta)]])
        return np.dot(m, point)