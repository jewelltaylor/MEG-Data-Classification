# MEG Data Classification
## Abstract
The following contains a detailed analysis and an overview of the methods used to generate a set of decoding time series pertaining to MEG data. The aim of a decoding analysis: “is to 'decode' different perceptual stimuli or cognitive states over time from dynamic brain activation patterns” (Grootswagers et al, 2017, p. 2). Due to the fact that MEG data is a specific type of neuroimaging time series data, a decoding analysis is well suited for data of its nature. In this particular example, the MEG data is collected from an individual over a single session that consisted of multiple trials. In each trial, the individual was shown a series of images in rapid succession. In this series of images, there was a target image. If the individual was able to correctly identify the presence of the target image, the corresponding MEG data was labeled correct. Conversely, if the individual was not able to correctly identify the presence of the target image, the corresponding MEG data was labeled incorrect. The MEG scan collects data from the individual from -385ms to 902ms from image onset. The purpose of this decoding analysis is to determine if it is possible to predict whether an individual identified the presence of an image based on MEG data as well as how the accuracy of this prediction over time and with different image sets. The following contains three sections: Introduction, Approaches and Next Steps. The introduction section will contain high level information about the data, preprocessing the data and evaluating the approaches. Next, in the approaches section, methods used to generate the decoded time series will be discussed and evaluated. Following this, considerations for future approaches will be considered in the next steps section. 

## Introduction
### Data
The MEG data initially consisted of a series of 24 folders containing distinct trial files. The 24 folders correspond to twelve image sets each with a folder with trial files with the correct label and trial files with the incorrect label. Trial files are matlab files that each contain a single matrix that has 323 rows and 1288 columns. The 1288 columns represent data from each millisecond over the time interval of the MEG scan for that trial. Of the 323 rows, only the first 306 are relevant data for this experiment. These 306 rows represent the features generated from the MEG scan. The objective of preprocessing in this experiment is to use the aforementioned MEG data to generate a set of 12 lists of 1288 datasets each corresponding to the observations of the MEG scan at each millisecond of the time interval from -385ms to 902ms after image onset. Each of the twelve lists corresponds to a particular image set. In this way, each of the 1288 datasets will consist of 307 columns that include 306 features columns and 1 label column.

### Class Distribution
In the context of this experiment, the data corresponding to each of the twelve image sets is imbalanced to varying degrees. Some instances have a large proportion of observations with correct labels and other instances have a large proportion of observations with incorrect labels. Each of the image sets has thirty or less observations. A table outlining in the distribution of the observations in each image set is below. 

<img width="621" alt="Screen Shot 2019-12-18 at 1 01 38 AM" src="https://user-images.githubusercontent.com/34798787/71060656-10144000-2134-11ea-832d-41d738b9f847.png">

In earlier iterations of approaches, a decoded time series was generated for each image set. Accordingly, twelve decoded time series were created in total. There were several challenges to these approaches that stemmed from the small amount of observation in each image set and the sometimes highly unbalanced nature of the data. As a result, in later iterations of approaches, observations from each image set were combined into an aggregate image set. The aggregate image set contains 88 correct observations and 238 incorrect observations. In these cases, a single decoded time series is generated because the observations corresponding to each image set are present in the aggregate dataset. For convenience, the two sets of aforementioned approaches will be referred to as Image Set Approaches and Aggregate Approaches, respectively. 

### Normalization 
An important thing to note is the extremely small nature of the values present in the MEG data. As such, the values of the 1288 datasets generated in the preprocessing steps are normalized. In this case, normalization consists of mapping every value in a dataset in the range of 0 to 1 based on the distribution of the initial values. In cases of datasets with extremely small values such as this, normalizing the datasets will make the subsequent models generated by the machine learning algorithms more accurate. 

### Evaluation
The error measure used to evaluate the performance of a model at each point in the time series is prediction accuracy. In order to obtain an accurate estimate of the prediction accuracy, K-Fold Cross Validation is used. In earlier iterations of approaches where decoded time series were generated for each image set, K-Fold cross validation is used, where n is the number of observations in the dataset. This is a special case of K-Fold cross validation called Leave-One-Out cross validation. LOO is helpful in cases where there is a limited number of observations in the dataset such as in the case of approaches generating a decoded time series for each individual image set. Conversely, in later iterations of approaches where observations from each image set were combined into an aggregate image set, 3-Fold cross validation is used. This is due a constraint in computational resources. In any event, 3-Fold cross validation will be enough to accurately estimates the models performance. 

### Approaches
In the following section, the various methods used to generate the decoded time series will be discussed and evaluated. In accordance to the above distinction, the approaches section will contain two main sections of approaches: Image Set Approaches and Aggregate Approaches. As mentioned previously, Image Set Approaches generate a set of twelve decoded time series whereas Aggregate Approaches generate a single decoded time series. For each approach in the following section, this information will be outlined: Decoded Time Series, Model, Methods and  Remarks. In the case of Image Set Approaches, only a single decoded time series will displayed for the sake of brevity. 

**Approach 1**

<img width="700" alt="Screen Shot 2019-12-18 at 1 50 34 AM" src="https://user-images.githubusercontent.com/34798787/71062566-d42fa980-2138-11ea-871f-1a0684d11096.png">

**Approach 2** 

<img width="700" alt="Screen Shot 2019-12-18 at 1 56 27 AM" src="https://user-images.githubusercontent.com/34798787/71062915-a9922080-2139-11ea-8222-878cd6de48b3.png">

**Approach 3**

<img width="700" alt="Screen Shot 2019-12-18 at 2 00 44 AM" src="https://user-images.githubusercontent.com/34798787/71063206-448afa80-213a-11ea-8c09-165fa8b006d8.png">

**Approach 4**

<img width="700" alt="Screen Shot 2019-12-18 at 2 03 35 AM" src="https://user-images.githubusercontent.com/34798787/71063423-a6e3fb00-213a-11ea-8700-903abde61db0.png">

**Approach 5**

<img width="700" alt="Screen Shot 2019-12-18 at 2 07 18 AM" src="https://user-images.githubusercontent.com/34798787/71063667-2f629b80-213b-11ea-95d6-f486bbaeff36.png">

**Approach 6**

<img width="700" alt="Screen Shot 2019-12-18 at 2 09 45 AM" src="https://user-images.githubusercontent.com/34798787/71063862-b3b51e80-213b-11ea-99fe-fef758df9125.png">

**Approach 7**

<img width="700" alt="Screen Shot 2019-12-18 at 2 13 59 AM" src="https://user-images.githubusercontent.com/34798787/71064106-40f87300-213c-11ea-87d9-5a30606e2fb2.png">

**Approach 8** 

<img width="700" alt="Screen Shot 2019-12-18 at 2 20 31 AM" src="https://user-images.githubusercontent.com/34798787/71064403-03481a00-213d-11ea-8e48-d43eecc7350e.png">

## Next Steps
Despite the limited success of the approaches outlined throughout the course of this report, this research has been extremely interesting and helpful in developing my skillset related to machine learning and computational neuroscience. I will continue to Iteratively revise my approach by applying my findings in each successive implementation until I can achieve successful results. After reflection, it is clear that the following points should be considered in subsequent implementations: 

-	Do the models generated in each individual image set have the potential to generalize to other image sets? In order to have access to more data, each series of image set datasets were combined into a single series of aggregated datasets. If the models don’t generalize outside of their image set, this is a bad idea.

-	Is the relationship between classes not separable by the learning algorithms (SVM, RF, NN) being used? Perhaps the learning algorithms being used are not sufficient for this classification problem. In this event, it would helpful to try other variations of classification algorithms such as K-Nearest Neighbors which leverages a geometric approach to classification. 

-	Are the abundance of features introducing too much noise into the models? It may be helpful to use feature selection and feature extraction methods in order to reduce the number of features in our data. This will reduce noise, computation time and lessen the effect of the cure of dimensionality. Note: This option has already been partially explored and applied to some of the approaches above.

-	Is there access to additional data or similar models elsewhere? If the current sample of data does not contain enough information for a learning algorithm to deduce a pattern, no amount of preprocessing, feature selection or model optimization will be able to produce a well performing model. The presence of other similar models could be a source of transfer learning for our model

## Exhibits
**Exhibit 1: Support Vector Machine Hyper Parameters**
<img width="700" alt="Screen Shot 2019-12-18 at 2 31 18 AM" src="https://user-images.githubusercontent.com/34798787/71065136-96358400-213e-11ea-9717-b876be19e825.png">



