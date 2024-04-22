import numpy as np
import matplotlib.pyplot as plt
import cv2

class KMeans:
    """
    KMeans clustering algorithm for clustering n-dimensional data.

    Attributes:
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations for the algorithm.
        centroids (list): List of centroids, initialized after first iteration.
    """
    def __init__(self, K=5, max_iters=100):
        self.K = K
        self.max_iters = max_iters
        self.centroids = []

    def predict(self, X):
        """
        Performs K-means clustering on the data X.

        Args:
            X (np.ndarray): The input data array of shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of cluster labels.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # Initialize centroids
        random_indices = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            clusters = self._create_clusters(self.centroids)
            new_centroids = self._calculate_centroids(clusters)
            
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

        return self._get_cluster_labels(clusters)

    def _create_clusters(self, centroids):
        distances = np.sqrt(((self.X - centroids[:, np.newaxis])**2).sum(axis=2))
        closest_centroids = np.argmin(distances, axis=0)
        return {i: np.where(closest_centroids == i)[0] for i in range(self.K)}

    def _calculate_centroids(self, clusters):
        return np.array([self.X[cluster].mean(axis=0) for cluster in clusters.values()])

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, samples in clusters.items():
            labels[samples] = cluster_idx
        return labels

def segment_image(image_path, K=5, max_iters=100, save_path='segmented_image.png'):
    """
    Segments an image using K-means clustering on the pixel values.

    Args:
        image_path (str): Path to the input image file.
        K (int): Number of desired clusters.
        max_iters (int): Maximum number of iterations for the K-means algorithm.
        save_path (str): Path where the segmented image will be saved.

    Returns:
        np.ndarray: An array representing the segmented image.
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Flatten image
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Clustering
    kmeans = KMeans(K=K, max_iters=max_iters)
    labels = kmeans.predict(pixel_values)
    
    # Reshape labels and convert to int type
    labels = labels.astype(int)
    segmented_image = labels.reshape(image.shape[:-1])
    
    # Map clusters to original image colors (average color of the cluster)
    masked_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(K):
        masked_image[segmented_image == i] = kmeans.centroids[i]

    # Convert back to uint8 and save
    masked_image = np.uint8(masked_image)
    plt.imsave(save_path, masked_image)
    
    return masked_image

# Usage example:
segmented_img = segment_image('images/1018.jpg', K=3, save_path='segmented_output.png')
plt.imshow(segmented_img)
plt.show()
