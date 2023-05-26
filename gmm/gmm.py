import time
import sklearn.mixture._gaussian_mixture as gmm
import pygame
import numpy as np

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class gmm_3d:
    def __init__(self, display):
        self.exit = False
        self.display = display
        self.cluster_count = config.getint('gmm3d', 'cluster_count')

        self.theta = 0.5
        self.phi = 0.5

        self.guess_means = np.zeros((self.cluster_count, 3))
        self.guess_covariances = np.zeros((self.cluster_count, 3, 3))

        self.reset()

    def reset(self):
        self.last_step = ''
        self.setup_gmm()
        # set the means to 3 random points in the data
        self.means = self.data[np.random.randint(0, self.data.shape[0], self.cluster_count), :]

        self.guess_means = np.zeros((self.cluster_count, 3))

    def update(self):
        while not self.exit:
            if pygame.mouse.get_pressed()[0]:
                rel = pygame.mouse.get_rel()

                if rel[1] < 10 or rel[0] < 10:
                    self.theta += rel[1] / 100
                    self.phi += rel[0] / 100

            for event in pygame.event.get():

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:
                        model = gmm.GaussianMixture(n_components=self.cluster_count, covariance_type='full', max_iter=1,
                                                    means_init=self.guess_means)
                        model.fit(self.data)
                        self.guess_means = model.means_
                        self.guess_covariances = model.covariances_
                        self.assignment = model.predict(self.data)
                        self.last_step = 'E-Step'

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
        self.draw_variances()
        self.show_info()
        pygame.display.flip()

    def draw_data(self):
        colors = [(255, 100, 50, 120), (50, 255, 100, 120), (100, 50, 255, 120), (255, 255, 100, 120),
                  (100, 255, 255, 120), (255, 100, 255, 120)]
        null_circle = pygame.Surface((10, 10), pygame.SRCALPHA)
        data_circle = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(null_circle, (120, 120, 120, 60), (5, 5), 5)
        tmpdata = self.data.copy()
        for i in range(self.data.shape[0]):
            tmpdata[i] = self.translate_in_3d(self.data[i])
        for i in range(self.data.shape[0]):

            if self.assignment[i] == -1:
                self.display.blit(null_circle, (int(tmpdata[i, 0] * 100) + 400 - 5,
                                                int(tmpdata[i, 1] * 100) + 300 - 5))

            else:
                pygame.draw.circle(data_circle, colors[int(self.assignment[i])], (5, 5), 5)
                self.display.blit(data_circle, (int(tmpdata[i, 0] * 100) + 400 - 5, int(tmpdata[i, 1] * 100) + 300 - 5))

    def draw_means(self):
        tmpmeans = self.guess_means.copy()
        for i in range(self.guess_means.shape[0]):
            tmpmeans[i] = self.translate_in_3d(self.guess_means[i])
        # draw a little cross at the position of the means
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

    def draw_variances(self):
        # this draws the variances of the gaussians around each mean
        # draw a circle around each mean with the radius of the variance, one circle for each dimension
        # scale these circles with the phi and theta values

        tmpmeans = self.guess_means.copy()
        for i in range(self.guess_means.shape[0]):
            tmpmeans[i] = self.translate_in_3d(self.guess_means[i])
        # draw a little cross at the position of the means
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

            # now draw the circles

            # first calculate the radius of the circle
            # this is the variance of the gaussian

            radius = np.sqrt(self.guess_covariances[i, 0, 0])
            # now draw the circle
            pygame.draw.circle(self.display, colors[i], (int(tmpmeans[i, 0] * 100) + 400,
                                                            int(tmpmeans[i, 1] * 100) + 300), int(radius * 100), 1)


    def setup_gmm(self):
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


        xvar = config.getfloat('gmm3d', 'x_variance')
        yvar = config.getfloat('gmm3d', 'y_variance')
        zvar = config.getfloat('gmm3d', 'z_variance')
        cov = np.array([[xvar, 0, 0], [0, yvar, 0], [0, 0, zvar]])
        self.means = np.zeros((self.cluster_count, 3))
        for i in range(self.cluster_count):
            self.means[i, :] = np.array([1.5 * np.cos(2 * np.pi * i / self.cluster_count),
                                         1.5 * np.sin(2 * np.pi * i / self.cluster_count),
                                         0])

        for i in range(self.cluster_count):
            self.data[i * 50:(i + 1) * 50, :] = np.random.multivariate_normal(self.means[i, :], cov, 50)

        for i in range(self.cluster_count):
            self.correct_assignment[i * 50:(i + 1) * 50, :] = i

        # self.iterate()

    def iterate(self):
        if self.last_step != 'estimation':
            self.last_step = 'estimation'
            # estimation step in gmm

            # calculate the new assignment
            for i in range(self.data.shape[0]):
                # calculate the probability of the data point to belong to each cluster
                prob = np.zeros((self.cluster_count, 1))
                for j in range(self.cluster_count):
                    prob[j, 0] = self.guess_weights[j] * self.get_gaussian_prob(self.data[i, :], self.means[j, :],
                                                                                self.guess_covs[j])
                # assign the data point to the cluster with the highest probability
                self.assignment[i, 0] = np.argmax(prob)


        else:
            self.last_step = 'maximization'
            # calculate the new means, covariances and weights
            # calculate the new means
            for i in range(self.cluster_count):
                # calculate the new mean for each cluster
                self.means[i, :] = np.sum(self.data[self.assignment[:, 0] == i, :], axis=0) / np.sum(
                    self.assignment[:, 0] == i)

            # calculate the new covariances
            for i in range(self.cluster_count):
                # calculate the new covariance for each cluster
                self.guess_covs[i, :, :] = np.dot(
                    np.transpose(self.data[self.assignment[:, 0] == i, :] - self.means[i, :]),
                    self.data[self.assignment[:, 0] == i, :] - self.means[i, :]) / np.sum(self.assignment[:, 0] == i)

            # calculate the new weights
            for i in range(self.cluster_count):
                # calculate the new weight for each cluster
                self.guess_weights[i, 0] = np.sum(self.assignment[:, 0] == i) / self.data.shape[0]

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

        colors = [(255, 100, 50, 120), (50, 255, 100, 120), (100, 50, 255, 120), (255, 255, 50, 120),
                  (50, 255, 255, 120), (255, 50, 255, 120)]

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
                      [-np.sin(self.phi), np.cos(self.phi) * np.sin(self.theta),
                       np.cos(self.phi) * np.cos(self.theta)]])
        return np.dot(m, point)

    def get_gaussian_prob(self, x, mean, cov):
        # calculate the probability of x given a mean and a covariance matrix
        # x is a 3d vector
        # mean is a 3d vector
        # cov is a 3x3 matrix
        # return a scalar
        print(cov)
        return 1 / np.sqrt(np.linalg.det(2 * np.pi * cov)) * np.exp(
            -0.5 * np.dot(np.dot(np.transpose(x - mean), np.linalg.inv(cov)), x - mean))
