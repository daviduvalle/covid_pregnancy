# Clustering of COVID-19 research papers related to pregnancy

This project tries to tackle the [Kaggle COVID-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558) Open Research Dataset Challenge task regarding risk factors. In particular, tries to shed light on the impact in pregnant women and neonates.

The problem is modeled as an unsupervised learning task with the goal to automatically find clusters of similar documents. Tf-idf and the K-means algorithm is used for cluster identification.

## Methodology
1. Each document's title and body are filtered looking for specific keywords such as *pregnant*, *pregnancy*, *neonatal*
2. The result set is preprocessed with common NLP methods. i.e. stop words removal, tokenization, lemmatization, etc.
3. Documents are then vectorized using Tf-idf and a matrix of (document x term) is generated
4. K-means uses the Tf-idf representation of the documents to find clusters, several artifacts are generated such as a final report mapping clusters to document sets and a 3-D scatter plot.

## Results

![](output/clustering_3d.png)

## Conclusions





