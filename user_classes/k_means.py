# import numpy as np
# import time
#
# def iterate(
#         data,
#         means,
#         assignment,
#         cluster_count,
#         draw=None):
#     '''
#     This function should be used to perform k-means clustering on a given dataset.
#     The data, means
#
#     :param data: (numpy array)
#     :param means: (numpy array)
#     :param assignment: (numpy array)
#     :param cluster_count: (int)
#     :param drawFunction: (function)
#
#     All of the above parameters can be used to store the state of the algorithm,
#     except for cluster_count, which is the number of clusters to use.
#
#
#     The draw function can be called to update the given display
#     with the current state of the algorithm.
#     '''
#
#
#     for j, data_point in enumerate(data):
#         # calculate the distance to each mean
#         distances = np.zeros(cluster_count)
#         for i in range(cluster_count):
#             distances[i] = np.linalg.norm(data_point - means[i, :])
#         # assign the data point to the index of the closest mean
#         if (assignment[j] != np.argmin(distances)):
#             assignment[j] = np.argmin(distances)
#             draw()
#             time.sleep(0.001)
#     last_step = 'maximization'
#     # calculate the new means
#     for i in range(cluster_count):
#         means[i, :] = np.mean(data[assignment[:, 0] == i, :], axis=0)
#     draw()
#
#     print("kmeans_user.py: iterate() called")
#
