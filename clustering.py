from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

custom_colors = {
    0: 'blue',
    1: 'orange',
    2: 'red',
    3: 'green',
    4: 'purple'  # Assuming you have a fifth cluster; add more if needed
}

def elbow_method(rfm_scaled):
# Determine the optimal number of clusters using the Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    return plt

def calculate_cluster_statistics(data):
    cluster_analysis = data.groupby('Cluster').agg({
        'Recency': ['mean', 'min', 'max'],
        'Frequency': ['mean', 'min', 'max'],
        'Monetary': ['mean', 'min', 'max']
    }).reset_index()

    cluster_analysis.columns = [' '.join(col).strip() for col in cluster_analysis.columns.values]
    return cluster_analysis

def plot_clusters(rfm_cleaned):
                  # Plotting the 3D scatter plots using Matplotlib
    fig = plt.figure(figsize=(20, 15))

    ax1 = fig.add_subplot(231, projection='3d')
    for cluster in rfm_cleaned['Cluster'].unique():
        cluster_data = rfm_cleaned[rfm_cleaned['Cluster'] == cluster]
        ax1.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'], label=f'Cluster {cluster}')
    ax1.set_title('Distribution of all instances coloured for different clusters')
    ax1.set_xlabel('Recency')
    ax1.set_ylabel('Frequency')
    ax1.set_zlabel('Monetary')
    ax1.legend()

    # Plot each cluster separately
    for i, cluster in enumerate(rfm_cleaned['Cluster'].unique()):
        ax = fig.add_subplot(232 + i, projection='3d')
        cluster_data = rfm_cleaned[rfm_cleaned['Cluster'] == cluster]
        ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'], label=f'Cluster {cluster}')
        ax.set_title(f'Distribution of the instances in cluster {cluster}')
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')

    return fig

def transaction_pie_chart(rfm_cleaned):
    transaction_counts = rfm_cleaned['Cluster'].value_counts()
    colors = [custom_colors[cluster] for cluster in transaction_counts.index]
    fig, ax1 = plt.subplots()
    ax1.pie(transaction_counts, labels=transaction_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Number of Transactions by Cluster')
    return fig

def monetary_pie_chart(rfm_cleaned):
    monetary_values = rfm_cleaned.groupby('Cluster')['Monetary'].sum()
    colors = [custom_colors[cluster] for cluster in monetary_values.index]
    fig, ax1 = plt.subplots()
    ax1.pie(monetary_values, labels=monetary_values.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Total Monetary Value by Cluster')
    return fig