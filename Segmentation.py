# class Segmentation:
#     def __init__(self, image):
#         self.image = image

#     def K_means(self):
#         pass

#     def MeanShift(self):
#         pass

#     def agglomerative_clustering(self):
#         pass

#     def region_growing(self):
#         pass


import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

def calculate_distance(cluster1, cluster2):
    # Calculate the Euclidean distance between two clusters in Luv color space
    return np.linalg.norm(cluster1 - cluster2)

def merge_clusters(cluster1, cluster2):
    # Merge two clusters by taking the mean of their values
    return (cluster1 + cluster2) / 2

def calculate_local_min_distances(clusters, start_idx, end_idx):
    # Calculate local minimum distances between clusters within a specified range
    local_min_distances = []
    for i in range(start_idx, end_idx):
        min_dist = float('inf')
        min_dist_idx = None
        for j in range(i+1, len(clusters)):
            dist = calculate_distance(clusters[i], clusters[j])
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = j
        local_min_distances.append((i, min_dist_idx, min_dist))
    return local_min_distances

def agglomerative_segmentation(image, num_clusters, num_threads=4):
    # Convert image to Luv color space
    image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    
    # Initialize clusters
    clusters = image_luv.reshape(-1, 3)
    
    # Multithreaded calculation of local minimum distances
    local_min_distances = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        chunk_size = len(clusters) // num_threads
        for i in range(0, len(clusters), chunk_size):
            end_idx = min(i + chunk_size, len(clusters))
            futures.append(executor.submit(calculate_local_min_distances, clusters, i, end_idx))
        for future in futures:
            local_min_distances.extend(future.result())

    # Perform agglomerative clustering
    while len(clusters) > num_clusters:
        # Find the global minimum distance from local minimum distances
        min_dist = float('inf')
        min_dist_indices = None
        for dist_info in local_min_distances:
            if dist_info[2] < min_dist:
                min_dist = dist_info[2]
                min_dist_indices = dist_info
        
        # Merge the two clusters with the minimum distance
        merged_cluster = merge_clusters(clusters[min_dist_indices[0]], clusters[min_dist_indices[1]])
        clusters[min_dist_indices[0]] = merged_cluster
        del clusters[min_dist_indices[1]]
        
        # Update local minimum distances
        new_local_min_distances = []
        for dist_info in local_min_distances:
            i, j, dist = dist_info
            if j != min_dist_indices[1]:
                if j > min_dist_indices[1]:
                    j -= 1
                new_local_min_distances.append((i, j, calculate_distance(clusters[i], clusters[j])))
        local_min_distances = new_local_min_distances

    # Reshape the clustered image
    segmented_image_luv = np.array(clusters, dtype=np.uint8).reshape(image_luv.shape)

    # Convert segmented image back to BGR color space
    segmented_image_bgr = cv2.cvtColor(segmented_image_luv, cv2.COLOR_Luv2BGR)

    return segmented_image_bgr

# Load the image
image = cv2.imread('images\Screenshot 2024-04-21 232935.png')

# Perform agglomerative segmentation with multithreading
num_clusters = 5
segmented_image = agglomerative_segmentation(image, num_clusters)

# Display original and segmented images
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
