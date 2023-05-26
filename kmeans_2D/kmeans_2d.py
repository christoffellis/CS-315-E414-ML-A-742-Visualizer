import time

import pygame
import numpy as np

class kmeans_2D:
    def __init__(self, display):
        self.exit = False
        self.display = display
        self.cluster_count = 3
        self.reset()

    def reset(self):
        self.last_step = ''
        self.setup_kmeans()
        # set the means to 3 random points in the data
        self.means = self.data[np.random.randint(0, self.data.shape[0], self.cluster_count), :]

    def update(self):
        while not self.exit:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.iterate()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.reset()


                if event.type == pygame.QUIT:
                    self.exit = True

            self.draw()

    def draw(self):
        self.display.fill((0, 0, 0))
        self.draw_plane()
        self.draw_data()
        self.draw_means()
        self.show_info()
        pygame.display.flip()

    def draw_plane(self):
        offset = (50, 50)
        # draw two lines in middle of screen
        pygame.draw.line(self.display, (255, 255, 255), (self.display.get_width() / 2, offset[1]),
                         (self.display.get_width() / 2, 550))
        pygame.draw.line(self.display, (255, 255, 255), (offset[0], self.display.get_height() / 2),
                            (self.display.get_width() - offset[0], self.display.get_height() / 2))

    def draw_data(self):
        colors = [(255, 100, 50), (50, 255, 100), (100, 50, 255)]
        null_color = (120, 120, 120)
        for i in range(self.data.shape[0]):
            if self.assignment[i, 0] == -1:
                pygame.draw.circle(self.display, null_color, (int(self.data[i, 0] * 100) + 400,
                                                              int(self.data[i, 1] * 100) + 300), 5)
            else:
                pygame.draw.circle(self.display, colors[
                    int(self.assignment[i, 0])
                ], (int(self.data[i, 0] * 100) + 400,
                                                                   int(self.data[i, 1] * 100) + 300), 5)

    def draw_means(self):
        #draw a little cross at the position of the means
        colors = [(255, 25, 12), (12, 255, 25), (25, 12, 255)]
        for i in range(self.means.shape[0]):
            pygame.draw.line(self.display, colors[i], (int(self.means[i, 0] * 100) + 400 - 5,
                                                       int(self.means[i, 1] * 100) + 300 - 5),
                             (int(self.means[i, 0] * 100) + 400 + 5,
                              int(self.means[i, 1] * 100) + 300 + 5), 2)
            pygame.draw.line(self.display, colors[i], (int(self.means[i, 0] * 100) + 400 - 5,
                                                       int(self.means[i, 1] * 100) + 300 + 5),
                             (int(self.means[i, 0] * 100) + 400 + 5,
                              int(self.means[i, 1] * 100) + 300 - 5), 2)

    def setup_kmeans(self):
        self.generate_data()

    def generate_data(self):
        # generate random 2d gaussian data
        # generate 50 points for 3 clusters
        self.data = np.zeros((50 * self.cluster_count, 2))
        self.assignment = np.zeros((50 * self.cluster_count, 1))
        self.assignment[:, 0] = -1
        self.correct_assignment = np.zeros((50 * self.cluster_count, 1))

        # set the means to 3 equidistant points on a circle
        # data clusters must have the same random cov, cov must be smaller than 0.5

        cov = np.array([[0.1, 0], [0, 0.1]])
        self.means = np.zeros((self.cluster_count, 2))
        for i in range(self.cluster_count):
            self.means[i, :] = np.array([np.cos(2 * np.pi * i / self.cluster_count),
                                         np.sin(2 * np.pi * i / self.cluster_count)])

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
            'Click for next step',
            'Current step: ' + self.last_step
        ]
        self.font = pygame.font.SysFont('arial', 20)
        for i, line in enumerate(text):
            self.display.blit(self.font.render(line, True, (255, 255, 255)), (50, 50 + i * 20))


