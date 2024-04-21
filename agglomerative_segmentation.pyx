# Save this code in a file with a .pyx extension, e.g., agglomerative_segmentation.pyx

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from libc.math cimport sqrt

cdef double calculate_distance(double[:] cluster1, double[:] cluster2):
    cdef double distance = 0.0
    cdef int i
    for i in range(3):
        distance += (cluster1[i] - cluster2[i]) ** 2
    return sqrt(distance)

cdef double[:] merge_clusters(double[:] cluster1, double[:] cluster2):
    cdef int i
    cdef double[:] merged_cluster = np.zeros(3, dtype=np.float64)
    for i in range(3):
        merged_cluster[i] = (cluster1[i] + cluster2[i]) / 2
    return merged_cluster

cpdef calculate_local_min_distances(double[:, :] clusters, int start_idx, int end_idx):
    cdef list local_min_distances = []
    cdef int i, j
    cdef double dist
    for i in range(start_idx, end_idx):
        min_dist = float('inf')
        min_dist_idx = -1
        for j in range(i + 1, clusters.shape[0]):
            dist = calculate_distance(clusters[i], clusters[j])
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = j
        local_min_distances.append((i, min_dist_idx, min_dist))
    return local_min_distances

cpdef agglomerative_segmentation(image, int num_clusters, int num_threads=4):
    image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    clusters = np.array(image_luv.reshape(-1, 3), dtype=np.float64)
    
    local_min_distances = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        chunk_size = clusters.shape[0] // num_threads
        for i in range(0, clusters.shape[0], chunk_size):
            end_idx = min(i + chunk_size, clusters.shape[0])
            futures.append(executor.submit(calculate_local_min_distances, clusters, i, end_idx))
        for future in futures:
            local_min_distances.extend(future.result())

    while clusters.shape[0] > num_clusters:
        min_dist = float('inf')
        min_dist_indices = None
        for dist_info in local_min_distances:
            if dist_info[2] < min_dist:
                min_dist = dist_info[2]
                min_dist_indices = dist_info
        
        merged_cluster = merge_clusters(clusters[min_dist_indices[0]], clusters[min_dist_indices[1]])
        clusters[min_dist_indices[0]] = merged_cluster
        clusters = np.delete(clusters, min_dist_indices[1], axis=0)
        
        new_local_min_distances = []
        for dist_info in local_min_distances:
            i, j, dist = dist_info
            if j != min_dist_indices[1]:
                if j > min_dist_indices[1]:
                    j -= 1
                new_local_min_distances.append((i, j, calculate_distance(clusters[i], clusters[j])))
        local_min_distances = new_local_min_distances

    segmented_image_luv = np.array(clusters, dtype=np.uint8).reshape(image_luv.shape)
    segmented_image_bgr = cv2.cvtColor(segmented_image_luv, cv2.COLOR_Luv2BGR)

    return segmented_image_bgr
