import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from tqdm import tqdm
from heapq import heappush, heappop

class Agglomerative_Clustering:
    def __init__(self, verbose=False, linkage_type='complete'):
        self.verbose= verbose
        self.linkage_type = linkage_type
        
    def min_value_not_on_diagonal(self, matrix):
        min_value = float('inf')
        min_x, min_y = -1, -1
        
        # Iterate through the elements of the matrix
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                # Check if the current element is not on the main diagonal
                if i != j:
                    # Update the minimum value and its indices if needed
                    if matrix[i][j] < min_value:
                        min_value = matrix[i][j]
                        min_x = i
                        min_y = j
        
        return min_value, min_x, min_y
        
    def cluster_distance(self, X, cluster_members):
        """
        Calculates the cluster distances based on the specified linkage type.

        Params
        ------
        cluster_members: dict
            Stores the cluster members in format: {key: [item1, item2 ..]}.
            If key is less than X.shape[0], then it only has itself in the cluster.

        Returns
        -------
        Distance: 2D array
            Contains distances between each cluster.
        """
        n_clusters = len(cluster_members)
        keys = list(cluster_members.keys())
        Distance = np.zeros((n_clusters, n_clusters))

        # Compute pairwise distances between clusters
        for i in range(n_clusters):
            ith_elems = cluster_members[keys[i]]
            for j in range(i + 1, n_clusters):
                jth_elems = cluster_members[keys[j]]
                d_in_clusters = euclidean_distances(X[ith_elems], X[jth_elems])
                if self.linkage_type == 'complete':
                    Distance[i, j] = np.max(d_in_clusters)
                elif self.linkage_type == 'single':
                    Distance[i, j] = np.min(d_in_clusters)
                # Since the distance matrix is symmetric, we can assign the same value to the opposite side
                Distance[j, i] = Distance[i, j]

        return Distance



    def fit(self, X):
        """
        Generates the dendrogram.

        Params
        ------
        X: Dataset, shape (nSamples, nFeatures)

        Returns
        -------
        Z: 2D array. shape (nSamples-1, 4). 
            Linkage matrix. Stores the merge information at each iteration.
        """
        self.nSamples = X.shape[0]
        cluster_keys = list(range(self.nSamples))
        cluster_members = {i: [i] for i in cluster_keys}
        Z = np.zeros((self.nSamples-1,4)) # c1, c2, d, count

        with tqdm(total=self.nSamples-1) as pbar:
            for i in range(0, self.nSamples-1):
                pbar.update(1)  # Increment progress bar by one step
                if self.verbose:
                    print(f'\n-------\nDebug Line at, i={i}\n--------')

                nClusters = len(cluster_members)
                keys = list(cluster_members.keys())
                # caculate the distance between existing clusters
                D = self.cluster_distance(X,cluster_members)
                
                # Using heap to find minimum value
                min_heap = []
                for x in range(len(D)):
                    for y in range(x+1, len(D)):
                        heappush(min_heap, (D[x, y], x, y))

                _, tmpx, tmpy = heappop(min_heap)

                if self.verbose:
                    print(f'Z:\n{Z}, \nCluster Members: {cluster_members}, D: \n {D}')

                x = keys[tmpx]
                y = keys[tmpy]
                # update Z
                Z[i,0] = x
                Z[i,1] = y
                Z[i,2] = D[tmpx, tmpy] # that's where the min value is
                Z[i,3] = len(cluster_members[x]) + len(cluster_members[y])

                # new cluster created
                cluster_members[i+self.nSamples] = cluster_members[x] + cluster_members[y]
                # remove merged from clusters pool, else they'll be recalculated
                del cluster_members[x]
                del cluster_members[y]

        self.Z = Z
        return self.Z

    
    def predict(self, num_cluster=3):
        """
        Get cluster label for specific cluster size.
        
        Params
        ------
        num_cluster: int. 
            Number of clusters to keep. Can not be > nSamples
        
        Returns
        -------
        labels: list.
            Cluster labels for each sample.
        """
        labels = np.zeros(self.nSamples, dtype=int)  # Initialize labels array
        cluster_members = {i: [i] for i in range(self.nSamples)}  # Initialize clusters
        
        # Iterate until desired number of clusters is reached
        for i in range(self.nSamples - num_cluster):
            x, y = int(self.Z[i, 0]), int(self.Z[i, 1])  # Get clusters to merge
            cluster_members[self.nSamples + i] = cluster_members[x] + cluster_members[y]  # Merge clusters
            del cluster_members[x]  # Remove merged clusters
            del cluster_members[y]
        
        # Assign labels to samples based on the final clusters
        for label, samples in enumerate(cluster_members.values()):
            labels[samples] = label
            
        return labels

def main():

    X,y = make_classification(100,n_features=2,n_redundant=0)
    print(X.shape)

    clusters = Agglomerative_Clustering(linkage_type='complete')
    Z = clusters.fit(X)
    Labels = clusters.predict(num_cluster=3)

    clustering = AgglomerativeClustering(n_clusters=3,linkage='complete').fit(X)
    skLabel = clustering.labels_

    fig, ax = plt.subplots(2,2,facecolor='white',figsize=(15,5*2),dpi=120)

    # Cluster
    for i in range(3):
        myIndices = Labels==i
        skIndices = skLabel==i
        ax[0,0].scatter(x=X[myIndices,0], y=X[myIndices,1],label=i)
        ax[0,1].scatter(x=X[skIndices,0], y=X[skIndices,1],label=i)
        
    ax[0,0].set_title('Custom | Cluster')
    ax[0,1].set_title('Sklearn | Cluster')
    ax[0,0].legend()
    ax[0,1].legend()

    # Dendrogram
    z = hierarchy.linkage(X, 'complete') # scipy agglomerative cluster
    hierarchy.dendrogram(Z, ax=ax[1,0]) # plotting mine with their function
    hierarchy.dendrogram(z, ax=ax[1,1]) # plotting their with their function

    ax[1,0].set_title('Custom | Dendrogram')
    ax[1,1].set_title('Sklearn | Dendrogram')
    plt.show()

if __name__ == '__main__':
    main()