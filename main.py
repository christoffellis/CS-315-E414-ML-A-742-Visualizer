import pygame

from gmm.gmm import gmm_3d
from kmeans_3D.kmeans_3d import kmeans_3D
from mnist.MNIST_visulzer import mnist_visualizer
from kmeans_2D.kmeans_2d import kmeans_2D
from dimension_reduction.dimension_reduction import dimension_reduction

from hmm.hmm import hmm_visualizer

mode = 'kmeans_3D'

from tkinter import *
from tkinter import ttk

options = ['dimension reduction', 'mnist - logis', 'kmeans_2D', 'kmeans_3D', 'GMM - TODO', 'hmm']
# create a tkinter window with each option
root = Tk()
root.title("Choose a mode")
root.geometry("400x280")
# set window position to center of user's screen
root.eval('tk::PlaceWindow . center')

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




if __name__ == "__main__":
    display = pygame.display.set_mode((800, 600))

    if mode == 'dimension reduction':
        dr = dimension_reduction(display)
        dr.update()

    if mode == 'mnist':
        visualizer = mnist_visualizer(display)
        visualizer.update()

    if mode == 'kmeans_2D':
        kmeans = kmeans_2D(display)
        kmeans.update()

    if mode == 'kmeans_3D':
        kmeans = kmeans_3D(display)
        kmeans.update()

    if mode == 'GMM - TODO':
        gmm = gmm_3d(display)
        gmm.update()


    if mode == 'hmm':
        hmm = hmm_visualizer(display)
        hmm.update()

