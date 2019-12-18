# MEG Data Classification
## Abstract
The following contains a detailed analysis and an overview of the methods used to generate a set of decoding time series pertaining to MEG data. The aim of a decoding analysis: “is to 'decode' different perceptual stimuli or cognitive states over time from dynamic brain activation patterns” (Grootswagers et al, 2017, p. 2). Due to the fact that MEG data is a specific type of neuroimaging time series data, a decoding analysis is well suited for data of its nature. In this particular example, the MEG data is collected from an individual over a single session that consisted of multiple trials. In each trial, the individual was shown a series of images in rapid succession. In this series of images, there was a target image. If the individual was able to correctly identify the presence of the target image, the corresponding MEG data was labeled correct. Conversely, if the individual was not able to correctly identify the presence of the target image, the corresponding MEG data was labeled incorrect. The MEG scan collects data from the individual from -385ms to 902ms from image onset. The purpose of this decoding analysis is to determine if it is possible to predict whether an individual identified the presence of an image based on MEG data as well as how the accuracy of this prediction over time and with different image sets. The following contains three sections: Introduction, Approaches and Next Steps. The introduction section will contain high level information about the data, preprocessing the data and evaluating the approaches. Next, in the approaches section, methods used to generate the decoded time series will be discussed and evaluated. Following this, considerations for future approaches will be considered in the next steps section. 

## Introduction
### Data
The MEG data initially consisted of a series of 24 folders containing distinct trial files. The 24 folders correspond to twelve image sets each with a folder with trial files with the correct label and trial files with the incorrect label. Trial files are matlab files that each contain a single matrix that has 323 rows and 1288 columns. The 1288 columns represent data from each millisecond over the time interval of the MEG scan for that trial. Of the 323 rows, only the first 306 are relevant data for this experiment. These 306 rows represent the features generated from the MEG scan. The objective of preprocessing in this experiment is to use the aforementioned MEG data to generate a set of 12 lists of 1288 datasets each corresponding to the observations of the MEG scan at each millisecond of the time interval from -385ms to 902ms after image onset. Each of the twelve lists corresponds to a particular image set. In this way, each of the 1288 datasets will consist of 307 columns that include 306 features columns and 1 label column.

### Class Distribution
In the context of this experiment, the data corresponding to each of the twelve image sets is imbalanced to varying degrees. Some instances have a large proportion of observations with correct labels and other instances have a large proportion of observations with incorrect labels. Each of the image sets has thirty or less observations. A table outlining in the distribution of the observations in each image set is below. 

![Class Distribution Table](https://ibb.co/F7ZDp6t)
