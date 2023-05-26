import ccxt
import pygame
import numpy
try:
    from user_classes import hmm

except ImportError:
    from hmmlearn import hmm
    print('User class hmm not found, using hmmlearn instead')
import numpy as np

class hmm_visualizer:
    def __init__(self, display):
        self.exit = False
        self.display = display
        self.data = None
        self.regionHeight = 25

    def reset(self):
        self.model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=50, random_state=42)
        self.model.fit(self.data.reshape(-1, 1))
        # Predict the hidden states corresponding to observed X.
        self.states = self.model.predict(self.data.reshape(-1, 1)[400:])

    def update(self):
        if self.data is None:
            text = 'Fetching data: USD/BTC'
            font = pygame.font.SysFont('Arial', 20)
            text_surface = font.render(text, True, (255, 255, 255))
            self.display.blit(text_surface, (self.display.get_width() / 2 - text_surface.get_width() / 2,
                                              self.display.get_height() / 2 - text_surface.get_height() / 2))
            pygame.display.flip()
            self.data = ccxt.binance().fetch_ohlcv('BTC/USDT', '1m', limit=1000)
            self.data = numpy.array(self.data)
            self.data = self.data[:, 3]
            avg = numpy.average(self.data)
            span = max(self.data) - min(self.data)

            self.data = (self.data - avg) / span * 1000

            # let each data point be equal to the difference between it and the previous data point
            tmpCopy = self.data.copy()
            for i in range(1, len(self.data)):
                self.data[i] = tmpCopy[i] - tmpCopy[i - 1]
            self.data[0] = 0


            self.reset()

        while not self.exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True

            # if mouse is between 100 and 125, increase bar height
            if pygame.mouse.get_pos()[1] > 100 and pygame.mouse.get_pos()[1] < 125:
                if pygame.mouse.get_pos()[0] > 100 and pygame.mouse.get_pos()[0] < 700:
                    self.regionHeight *= 1.2
                    self.regionHeight = min(self.regionHeight, 400)
                    self.reset()
            else:
                self.regionHeight *= 0.9
                self.regionHeight = max(self.regionHeight, 25)
                self.reset()

            self.draw()
            pygame.display.flip()

    def draw(self):
        self.display.fill((0, 0, 0))
        for i in range(400, len(self.data) - 1):
            c = (255, 100, 25) if self.data[i] > self.data[i + 1] else (100, 255, 25)
            pygame.draw.line(self.display, c, ( (i - 400)  + 100, max(min(self.data[i] + 300, 500), 100)),
                             ((i - 400)  + 101, min(max(self.data[i + 1] + 300, 100), 500)), 1)


        colors = [(255, 100, 50, 50), (50, 255, 100, 50), (100, 50, 255, 50)]
        for i, state in enumerate(self.states):
            surface = pygame.Surface((1, self.regionHeight), pygame.SRCALPHA)
            pygame.draw.rect(surface, colors[state], (0, 0, 1, self.regionHeight), 0)
            self.display.blit(surface, (i + 100, 100))
        pygame.draw.rect(self.display, (255, 255, 255), (100, 100, 600, 400), 1, border_radius=4)

        self.draw_info()

    def draw_info(self):
        cov = self.model.covars_

        text = [
            "Low Volatility",
            "Medium Volatility",
            "High Volatility"
        ]

        font = pygame.font.SysFont('Arial', 20)
        # let the colors list equate to the covariance matrix.
        # Red corrosponds to high volatility, green to medium, and blue to low
        colors = [(255, 75, 25), (25, 255, 75), (75, 25, 255)]
        # Sort the colors matrix in the same order as the covariance matrix
        for i in range(3):
            for j in range(3):
                if cov[i] == self.model.covars_[j]:
                    colors[i], colors[j] = colors[j], colors[i]
                    text[i], text[j] = text[j], text[i]

        for i in range(3):

            text_surface = font.render(text[i], True, colors[i])
            self.display.blit(text_surface, (self.display.get_width() / 2 - text_surface.get_width() / 2 + 200 * (i-1),
                                              50))

        mostLikelyNextState = self.model.transmat_[self.states[-1]].argmax()
        text_surface = font.render("Most likely next state: " + text[mostLikelyNextState], True, colors[mostLikelyNextState])
        self.display.blit(text_surface, (self.display.get_width() / 2 - text_surface.get_width() / 2,
                                            550))



