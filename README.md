# Clustering of COVID-19 research papers related to pregnancy

This project tries to tackle the [Kaggle COVID-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558) Open Research Dataset Challenge task regarding risk factors. In particular, tries to shed light on the impact in pregnant women and neonates.

The problem is modeled as an unsupervised learning task with the goal to automatically find clusters of similar documents. Tf-idf is used for featurization and the K-means algorithm is used for cluster identification.

## Methodology
1. Each document's title and body are filtered looking for specific keywords such as *pregnant*, *pregnancy*, *neonatal*
2. The result set is preprocessed with common NLP methods. i.e. stop words removal, tokenization, lemmatization, etc.
3. Documents are then vectorized using Tf-idf and a matrix of (document x term) is generated
4. K-means uses the Tf-idf representation of the documents to find clusters, several artifacts are generated such as a final report mapping clusters to document sets and a 3-D scatter plot.

## Results

![](output/clustering_3d.png)

17 clusters were identified using the elbow method. These are the clusters sorted (descending) by number of documents

    use study sample sequence case, 395
    cells mice virus use infection, 289
    patients study respiratory children rsv, 138
    cells mice cd il microglia, 103
    health public countries disease diseases, 90 
    piglets pig pdcov pedv pcv, 77
    zikv cells denv mice infection, 71
    pedv pig piglets strain sow, 70 
    influenza sari patients hn pandemic, 67 
    et al cells study use, 61 
    vaccine vaccines nanoparticles use virus, 57 
    ifn prrsv cells ifns type, 46
    calve herd calf dairy brsv, 45
    bat species alecto sequence use, 30 
    igy ha brv use pnsia, 28
    pneumonia children case lus study, 24
    ceacam cea ceacams genes domain, 13
    
A JSON report containing a mapping between clusters and document titles can be found [here](output/final_report.json)

The data didn't contain links to publication so titles need to be searched on a search engine to access the full documents.
    
## Conclusions

* Only 1,604 documents contained pregnancy related keywords out of 33,375. This is only 4.8% of all the docs in the dataset.
* Not all the research papers in the dataset are specific to Covid-19. This is a sample from the cluster "cells mice virus use infection", title: **A Replicating Modified Vaccinia Tiantan Strain Expressing an Avian-Derived Influenza H5N1 Hemagglutinin Induce Broadly Neutralizing Antibodies and Cross-Clade Protective Immunity in Mice** 
* The clusters with least documents contain research regarding pneumonia in children and a cell adhesion molecule [CEACAM1](https://en.wikipedia.org/wiki/CEACAM1) that seems to modulate innate immune responses. Probably both good research areas.

## How to run
    # Make sure you are using python3.6 or later
    git clone https://github.com/daviduvalle/covid_pregnancy.git 
    cd covid_pregnancy 
    # Create a virtual env and switch to it
    virtualenv -p python3 env
    source env/bin/activate
    pip install -r requirements.txt
    # Make sure the dataset is unzipped inside the covid/ directory
    python3 kmeans_runner.py
    
