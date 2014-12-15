Kareem Francis

Naive Bayes Article Classifier

5/27/12

CS363 Artificial Intelligence

CUNY Queens College

###Requirements:
* Python 2.7
* NLTK 2.0.1, which requires PyYAML 3.09 and Numpy 1.6.1
* Beautiful Soup is requires for the web scraper

###Execution:
From the project source directory, run 
python nbac.py [-h] [-training_dir TRAINING_DIR] [-testing_dir TESTING_DIR]
               [-o OUTPUT] [-c] [-v]

###Naive Bayes Article Classifier

####optional arguments:
 + -h, --help           
        show this help message and exit
 + -training_dir TRAINING_DIR
        A directory containing training articles and a class.txt file.
 + -testing_dir TESTING_DIR
        A directory containing test articles and a class.txt file
 + -o OUTPUT, --output OUTPUT
        The file to which output is to be written. Defaults to predictions.txt in current directory
 + -c, --clean
        Execute without using previous data. Necessary if training data has changed
 + -v, --verbose         
        Execute in verbose mode. Greatly increases printed output.

If no training or testing directories are specified, the default locations are used, which are located in the root of the project folder. The default location of the output file is the same directory as the python modules. Important: even though both the training and testing folders contain the same articles, only those files listed in class.txt will be opened, and there is NO overlap between the two.

###Overview:
Naive Bayes classification is a probabilistic approach to classifying/categorizing information based on provided information. It is considered naive because the algorithm ignores dependences between features, assuming them to be independent.
A document classifier is one that labels documents - in this case, news articles - in to predefined categories based on their content. A naive Bayes document classifier applies naive Bayes to the task of document classification.
Clustering, unlike classification is an unsupervised machine learning technique, which means that the input labels (classifications) are unknown. Data is grouped by similarities, but those similarities are unknown prior execution, and must be discerned.
K-Means is a clustering algorithm that groups data into K clusters, with each data entry being a member of the cluster to which it is the closest. There are many ways in which the distance between data points can be measured, such as Euclidian distance and cosine similarity, which measures the angle between vectors.

###Files:
1. README.rtf - this file
2. Project Overview.rtf - a description of the approach I used
3. Newscraper.py - the web scraper I wrote to gather news articles from Google News
4. NBAC.py - the main program, which performs both classification and clustering
5. NBAC_Ensemble.py – my attempt at an ensemble classification approach
6. Article.py - a class used as the data representation of an article
7. Model.p and Data.p - serialized execution data, used to avoid having to recompute data between executions, which may be time and processor intensive.
8. Predictions.txt: contains output of classification of test documents 
9. Clusters.txt: contains output of clustering of test documents
10. Performance Report.rtf: contains analysis of program results

