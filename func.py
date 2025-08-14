import numpy as np
import os
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def kmeans(data, k, max_iters=100):
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    labels = np.zeros(n_samples)
    for _ in range(max_iters):
        for i in range(n_samples):
            distances = [euclidean_distance(data[i], centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

def compute_inertia(data, labels, centroids):
    inertia = 0
    for i in range(len(data)):
        inertia += euclidean_distance(data[i], centroids[int(labels[i])]) ** 2
    return inertia

def elbow_method(data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        labels, centroids = kmeans(data, k)
        inertia = compute_inertia(data, labels, centroids)
        inertias.append(inertia)
    plt.figure()
    plt.plot(range(1, max_k + 1), inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method Showing the Optimal k')
    plt.show()

def silhouette_score(data, labels):
    n_samples = len(data)
    A = np.zeros(n_samples)
    B = np.zeros(n_samples)
    for i in range(n_samples):
        same_cluster = data[labels == labels[i]]
        other_clusters = data[labels != labels[i]]
        A[i] = np.mean([euclidean_distance(data[i], point) for point in same_cluster if not np.array_equal(data[i], point)])
        B[i] = np.mean([euclidean_distance(data[i], point) for point in other_clusters])
    S = (B - A) / np.maximum(A, B)
    return np.mean(S)

def pca(data, n_components=2):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]
    return np.dot(centered_data, sorted_eigenvectors[:, :n_components])

def plot_clusters(data, labels, title="Cluster Visualization"):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.show()

def cluster_task_1(sample):
    optimal_k = 2  
    labels, centroids = kmeans(sample, optimal_k)
    plot_clusters(sample, labels, title="Cluster Visualization for Task 1")
    silhouette_avg = silhouette_score(sample, labels)
    print(f'Silhouette Score for cluster_task_1: {silhouette_avg}')
    return labels

def cluster_task_2(sample):
    elbow_method(sample)
    optimal_k = 4  
    labels, centroids = kmeans(sample, optimal_k)
    plot_clusters(sample, labels, title="Cluster Visualization for Task 2")
    silhouette_avg = silhouette_score(sample, labels)
    print(f'Silhouette Score for cluster_task_2: {silhouette_avg}')
    return labels

def cluster_task_3(sample):
    optimal_k = 3  
    labels, centroids = kmeans(sample, optimal_k)
    plot_clusters(sample, labels, title="Cluster Visualization for Task 3 (original dimensions)")
    reduced_data = pca(sample)
    plot_clusters(reduced_data, labels, title="Cluster Visualization for Task 3 (PCA)")
    silhouette_avg = silhouette_score(sample, labels)
    print(f'Silhouette Score for cluster_task_3: {silhouette_avg}')
    return labels

def cluster_task_4(sample):
    elbow_method(sample)
    optimal_k = 3  
    labels, centroids = kmeans(sample, optimal_k)
    plot_clusters(sample, labels, title="Cluster Visualization for Task 4 (original dimensions)")
    reduced_data = pca(sample)
    plot_clusters(reduced_data, labels, title="Cluster Visualization for Task 4 (PCA)")
    silhouette_avg = silhouette_score(sample, labels)
    print(f'Silhouette Score for cluster_task_4: {silhouette_avg}')
    return labels

def main():
    data1 = np.load(os.path.join("data", "data1.npy"))
    data2 = np.load(os.path.join("data", "data2.npy"))
    data3 = np.load(os.path.join("data", "data3.npy"))
    data4 = np.load(os.path.join("data", "data4.npy"))

    label1 = cluster_task_1(data1)
    label2 = cluster_task_2(data2)
    label3 = cluster_task_3(data3)
    label4 = cluster_task_4(data4)

    np.save(os.path.join("output", "label1.npy"), label1)
    np.save(os.path.join("output", "label2.npy"), label2)
    np.save(os.path.join("output", "label3.npy"), label3)
    np.save(os.path.join("output", "label4.npy"), label4)

if __name__ == '__main__':
    main()   