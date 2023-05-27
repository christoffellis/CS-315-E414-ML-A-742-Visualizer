import pygame

from gmm.gmm import gmm_3d
from gradient_descent.gradient_descent import gradient_descent_visualizer
from kmeans_3D.kmeans_3d import kmeans_3D
from mnist.MNIST_visulzer import mnist_visualizer
from kmeans_2D.kmeans_2d import kmeans_2D
from dimension_reduction.dimension_reduction import dimension_reduction

from hmm.hmm import hmm_visualizer

mode = 'kmeans_3D'


from tkinter import *
from tkinter import ttk

options = ['Dimension Reduction', 'MNIST - Logistic Regression', 'Gradient Descent', 'K-Means 2D', 'K-Means 3D', 'Gaussian Mixture Models', 'Hidden Markov Models']
# create a tkinter window with each option
root = Tk()
root.title("CS/ML Visualizer")
root.geometry("400x360")
# set window position to center of user's screen
root.eval('tk::PlaceWindow . center')

# remove borders
#root.overrideredirect(True)

root.resizable(False, False)

# create a label for the tkinter window in the center of the window
label = ttk.Label(root, text="Choose a mode", font=("Arial", 20))
label.grid(column=0, row=0, padx=100, pady=36)

# resize the tkinter window to fit the label
root.update()

def set_mode(option):
    global mode
    mode = option
    root.destroy()

# create a button for each option
for i in range(len(options)):
    button = ttk.Button(root, text=options[i], command=lambda i=i: set_mode(options[i]))
    button.grid(column=0, row=i+1)

root.mainloop()

#mode = 'Gradient Descent'


if __name__ == "__main__":
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CS/ML Visualizer - " + mode)

    if mode == 'Dimension Reduction (WIP)':
        dr = dimension_reduction(display)
        dr.update()

    if mode == 'MNIST - Logistic Regression':
        visualizer = mnist_visualizer(display)
        visualizer.update()

    if mode == 'Gradient Descent':
        gradient_descent = gradient_descent_visualizer(display)
        gradient_descent.update()

    if mode == 'K-Means 2D':
        kmeans = kmeans_2D(display)
        kmeans.update()

    if mode == 'K-Means 3D':
        kmeans = kmeans_3D(display)
        kmeans.update()

    if mode == 'Gaussian Mixture Models':
        gmm = gmm_3d(display)
        gmm.update()


    if mode == 'Hidden Markov Models':
        hmm = hmm_visualizer(display)
        hmm.update()

