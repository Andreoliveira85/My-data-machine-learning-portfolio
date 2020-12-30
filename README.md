# Table of contents
- [Project 1. Sentiment-analysis-of-Tweets-and-prediction-of-bitcoin-prices](#P1)
- [Project 2. NYSE Prediction of Prices](#P2)
- [Project 3. Electronic Signature of Loans](#P3)
- [Project 4. Hybrid Mutual Fund Analysis](#P4)
- [Project 5. Linear Regression in E-commerce](#P5)
- [Project 6. Advertisement with ML](#P6)
- [Project 7. Detecting fraud with ML](#P7)
- [Project 8. NLP MiniProject](#P8)
- [Project 9. Classification of galaxies, stars and quasars](#P9)
- [Project 10. Miniproject DL: Breast cancer detection](#P10)
- [Project 11.DL project: Yolo object detection](#P11)
- [Project 12. Tinder recommendation system](#P12)
- [Project 13. Clustering for Uber Pickups](#P13)
- [Project 14. NLP-Project-with-Deep-Learning: Construction of a Translator machine](#P14)
- [My first data science project:Anomalies of temperatures in RJ and rainfalls of SP](#P15)
  
# [Project 1.Sentiment-analysis-of-Tweets-and-prediction-of-bitcoin-prices](https://github.com/Andreoliveira85/Sentiment-analysis-of-Tweets-and-prediction-of-bitcoin-prices) <a name="P1"></a>
**Final project for concluding the Fullstack formation in Data Science at Jedha in Paris.**

## Project overview:

* Firstly we scrapped 13 institutional and private Twitter accounts using Twitter APIs from April 2018 to mid September 2020.
* We used historical datasets of Bitcoin and other top coins in the crypto space in the same time period.
* We performed an initial exploratory data analysis on the coins historical datasets in order to understand the dominance of BTC in relation to other coins and the evolution of prices in the recent years.
* We merged the twitter datasets and we runned a sentiment analysis on those approximately 60000 tweets. The main reactions to BTC were positive during that period.
* Using the index of polarity built on the tweets and the close prices of BTC in the previous day we built a LSTM model with several dense layers and a stop on the final bias. The neural network performed  well in predicting prices both on the train set (April 2018- April 2020) with a mean square error app. 2.25% and on the test set (May 2020-14 Sept 2020) with a mean square error of 3.89%. 
* We gathered the predictions from the LSTM model and we used them as regressors for the FB Prophet Algorithm in order to predict prices from the end of the timeset (15 Sept 2020) until 30 November 2020. We used this method as a validation/test for the robustness of the predictions output by our LSTM model. The prices were close of the real prices gathered on google Finance for this new time period but only in a matter of 2 weeks (beginning October). This can be explained empirically by the big noise component of the time series data that reflects the bubble "crescendo" unexpected trend of BTC during the last three months.
* As a final note we reflect about the difficulty of predicting 1 day forward returns for BTC prices (multi-step model) due to the high volatility of the series that when managed creates a high bias on the architecture of the model. As a future project we will come back to this point.


## Code and resources used:
**Python version:** 3.7
**Packages/Libraries:** pandas, numpy, plottly, seaborn, matplotlib, tweepy, prophet, tensorflow, keras, scikit-learn, textblob, wordcloud.
**Datasets involved:** kaggle historical datasets on prices of BTC and other top coins, datasets of tweets created by several scrapped tweet accounts. 

### Sidenotes: 
The description of the datasets used is done on the slides of the [final presentation](https://github.com/Andreoliveira85/Sentiment-analysis-of-Tweets-and-prediction-of-bitcoin-prices/blob/main/JEDHA_FINAL_PROJECT_FULLSTACK_FINAL_PRESENTATION-1-14.pdf) . We could not upload the datasets here due to its size. We invite the visitor to check the graphics and visualizations created on the notebooks displayed in the final presentation.

* The following chart indicates the distribution of volume of the main crypto currencies. We learned the predominance of the theter coin over BTC. Theter is widely used on East Asia as a way of transaction due to its parity close to 1USD.

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/pieplot_currencies.png)

* The following boxplot indicates the volume of the coins. 

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/currencies_boxplot.png)

* The following scatter plot (log scale to overcome the high skewness of the cloud around BTC) shows the predominance of BTC followed by the other coins when considering the variables close price and volume in the crypto space.

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/scatter_plot_volume_close_log.png)

* The following graph represents the evolution of close prices of BTC over the last years. We remark the meteoric growth from 2016 to 2017 followed by a steep decline. During this year of 2020 we observe the highly fast increasing trend of the coin and we can observe around the 1st trimester the influence of the covid pandemic crisis on the prices.

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/bitcoin_prices.png)

* The following plot represents the sentiment analysis on the tweets that we performed over the 13 different instiutional and private accounts.
The tendency of the index that we built on the classification of the tweets is positive on BTC.

![my_image](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/sentiment_analysis_tweets2.jpg) 
 
 
* Split between train and test set:


* split <img src="https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/training_test_dataset.jpg" />



* The results of the LSTM model on the train set (April 2018-April 2020 approx.). The model learned quite well.

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/model_training_set.jpg). 

* The results of our model on the test set (April 2020- mid Sept 2020). The trends are followed by the predictions. There is a time period where the model performs far worst than the actual prices (August-September) but then the actual prices and the predictions start to converge again in the last two weeks of the dataset almost coallescing in the end.

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/model_test_dataset.jpg)

* The loss function (Mean squared error) on train (blue) and test (orange) set (per epoch). Scale 10^-3

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/loss_function.jpg)

* About the 1 day forward returns of the BTC:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/returns_new_model_train_set.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/returns_test_set_new_model.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/fig2.jpg)



* The predictions built by our model were used as regressors to feed the FB Prophet algorithm. The seasonality effects of the predictions based on the LSTM predictors are shown below. Monthly and bigger size trends should be neglected since the coin is not mature enough for such extrapolations. Although interesting to note the weekly trend of the price going down on Thusrdays and going up during the weekends.

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/prophet_model2.jpg)

* The results of the Prophetization of our predictions:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/prophet_model.jpg)

* The contrast of the predictions of the FB Prophet model feeded by the outputs of the LSTM net we built with the actual prices from 15 Sept 2020 until 30 November 2020 (fetched on Google Finance)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/prophets_results_reality.jpg)

**Conclusion:** the predictions of our LSTM model when are used as predictors in the FB Prophet algorithm produce new predictions that are close of the close actual value of the BTC according to Google Finance during a couple days (15 Sept until beg October). After that the actual value takes off much more than the predictions. This can be empirically understood due to the big component of noise in this time series and the recent bubble effect registered with BTC in the crypto space that is steeper than the high derivative of prices in 2017. According to this model this would be a good time to sell.


# [Project 2. NYSE Prediction of Prices](https://github.com/Andreoliveira85/NYSE-prediction-prices)  <a name="P2"></a>
Deep Learning project

## Project overview:

We use two datasets relative to historical data for stocks market at the New York Stock Exchange market for the year 2016. We performed an exploratory data analysis of those stocks and created a neural net mix of GRU with LSTM models to predict prices. The predictions and the real values follow the same trends of growth or ungrowth genreally speaking and around certain periods of time (before the 50th day of that year and around day 130) they almost coalesce. 

* Performance of the algorithm:
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/NYSE_prices.jpg)

## Code and resources used:
**Python version:** 3.7
**Packages:libraries:** numpy, pandas, matplotlib, scikitlearn, math, keras.
**Datasets:** available at Kaggle



# [Project 3. Electronic Signature of Loans](https://github.com/Andreoliveira85/electronic-signature-loans)  <a name="P3"></a>
## Project description: 

Confrontation of several Machine Learning algorithms to predict (classification) electronic signature of contracts. The dataset is a collection of financial info from clients  of a  private anonymous firm. The ML algorithms used were ANN, support vector machines and ensemble learning techniques: random forest classifiers, gradient boosting and adaboost. The algorithms with best score were our neural netowork and the random forest clasifier. 

## Code and resources used:
**Python version:** 3.7
**Packages:libraries:** numpy, pandas, matplotlib, scikitlearn, math, keras.
**Datasets:** dataset from anonymous private company available in the repo.



# [Project 4. Hybrid Mutual Fund Analysis](https://github.com/Andreoliveira85/Hybrid-Mutual-fund-Analysis)   <a name="P4"></a>
## Project overview: 
### Aim (Exploratory Data Analysis Project):
  * Analyse various parameters related to the Hybrid Mutual fund dataset and find distinction between good and bad schemes.
  
  * A hybrid fund is an investment fund that is characterized by diversification among two or more asset classes. These funds typically invest in a mix of stocks and bonds. The term hybrid indicates that the fund strategy includes investment in multiple asset classes. These funds offer investors an option for investing in multiple asset classes through a single fund. These funds can offer varying levels of risk tolerance ranging from conservative to moderate and aggressive. We carry a thorough exploratory visualization analysis of a dataset of funds traded by a firm in order to identify bad/good schemes according to several criteria. The exploration is done by multivariate analysis of the different variables of the dataset.

## Preview of some visualizations:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid1.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid2.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid3.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid4.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid5.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid6.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid7.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid8.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid9.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid10.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid11.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid12.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid13.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid14.jpg)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hybrid15.jpg)



## Code and resources used:
**Python version:** 3.7
**Packages:libraries:** numpy, pandas, matplotlib, seaborn.
**Datasets:** available at the repo




# [Project 5. Linear Regression in E-commerce](https://github.com/Andreoliveira85/Project-Linear-regression-in-E-commerce)   <a name="P5"></a>

# Project description:

Some Ecommerce company based in New York City sells clothing online but also has in-store style and clothing advice sessions. 
Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app 
or website for the clothes they want. The company is trying to decide whether to focus their efforts on their mobile app experience or their website. We use linear 
regression in order to quantify the linear dependence of the target variable "yearly amount spent" that captures the investment done in terms of the other variables
of the dataset. The score (Adjusted R2) is incredibly high 98% in the train and test set (with 20% size for the validation dataset). We also perform regularization
of the model (that in this case is not highly required since there is no overfitting in the model). Lasso regularization performs for the hyper parameter 1 a slight 
underfitting (Score on the train set : 0.981235530537366 vs Score on the test set : 0.9787641440205315) and the best performance for Ridge regularization on the model
is the vanishing coefficient 0 reducing it to the clasical linear regression model that we built in the first try.


## Some visualizations:

* Bivariate EDA

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/linear_model_yearly_amount_spent_vs_time_website.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/linear_model2.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/linear_model3.jpg)

* Empirical Evidence supporting linear regression:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/linear_model_evidence_regression.jpg)

* Gaussian residuals of the linear model:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/linear_model_gaussian_residuals.jpg)

* Coefficients of the linear model:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/linear_model_coeff.jpg)

## Code and resources used:
**Python version:** 3.7

**Libraries/packages used:** pandas, numpy, matplotlib, seaborn, scikitlearn.

**Dataset:** available at the repo.




# [Project 6. Advertisement with ML](https://github.com/Andreoliveira85/Advertisement-with-ML)   <a name="P6"></a>
## Project description:
In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user (classification problem).

This data set contains the following features:

* 'Daily Time Spent on Site': consumer time on site in minutes
* 'Age': cutomer age in years
* 'Area Income': Avg. Income of geographical area of consumer
* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
* 'Ad Topic Line': Headline of the advertisement
* 'City': City of consumer
* 'Male': Whether or not consumer was male
* 'Country': Country of consumer
* 'Timestamp': Time at which consumer clicked on Ad or closed window
* 'Clicked on Ad': 0 or 1 indicated clicking on Ad

## Methodologies: 

We approach the classification problem of predicting the variables 'click on the ad' by the visitors of the website using different classification
algorithms: logistic regression and random forest classifiers. After initial EDA we construct a ML pipeline where the split of the dataset is done with 34% test size. The logistic model does not see any under or over fitting of data. The random forest classifier pergorms better (96% score) in the validation test set
than the logistic model(90%). Using the random forest model we create a hiearchy of features importance. We conclude that the variable more influencial to predict
clicks on the ad is the daily internet usage of the user that visits the website. Since this is a variable that we can not control we decided to run the random forest model where this variable is erased from the ensemble of explanatory features. The score of the model goes a bit down when we run this version of the algorithm. Although interestingly we register in the hierachy of the new features importance that the variable are income surpasses the varaible age (which does not happen in the features importance of the first model). The third part of this project concerns hypothesis A/B testing where we analyse different confidence intervals (5% and 40% associated risk levels respectively) for the proportion rate of clicks done by male and female visitors. For the smallest level of risk 5% nothing can be concluded. Although for the higher level of 40% risk we observe empirically by random sampling thar the proportion of clicks on the add done by females surpasses the one done by males. This can be seen as a first indication of a marketing future strategy to be implemented. Although the Qui2 test built with 5% risk tell us that the variables "click on the add" and "gender" are independent concluding this discussion.  

# Visualizations:

* Multivariate EDA: Age vs area income

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/adv_area_income_age.jpg)

* Multivariate EDA: Age vs daily time spent on site

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/daily_time_site_age.jpg)

* Multivariate EDA: Daily internet usage vs daily time spent on site

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/daily_time_site_daily_int_usage.jpg)

* Classification report of the logistic regression model on the test validation set:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/classification_report_logistic_model.jpg)

*  Classification report of the random forest classifier on the test validation set:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/classification_report_rf.jpg)


* Feature importance hierarchy using the random forest classifier:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/feature_imp1.jpg)

*  Feature importance hierarchy running the random forest classifier with deprecation of the variable "daily internet usage of the user":

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/feature_imp_2.jpg)

* Empirical click rate on the add per gender:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/empirical_click_rate.jpg)

* Hypothesis testing for the click rates with 5% risk on the confidence level:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hyp_test1.jpg)

* * Hypothesis testing for the click rates with 40% risk on the confidence level:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/hyp_testing2.jpg)
 



 




## Code and resources used:
**Python version:** 3.7
**Libraries/packages used:** pandas, numpy, matplotlib, seaborn, scikitlearn
**Dataset:** available at the repo




# [Project 7. Detecting fraud with ML](https://github.com/Andreoliveira85/Detecting-fraud-with-ML)  <a name="P7"></a>
## Project description: 

We use a dataset from a private anonymous company in order to classify fraudulent clients taking other variables such as age, gender, browser used to shop, among 
others as explnatory variables.

## Methodologies:

After cleaning the dataset we start with a exploratory visualisation analysis of the data (univariate and multi variate). We register 9.4% of fraudulent clients on the dataset. After some feature engineering pipeline specifically built to handle the difference between the cathegorical and the numerical explanatory variables (Kbins discretizers from scikitlearn) we built a Bernoulli naive Bayes model for a split of 30% between train and test sets on the data. The Naive Bayes model performw well with a confusion matrix fully charged detecting all the classes for this unbalanced dataset. The Naive Bayes model predicts a rate of fraud of 9.36 % (the empirical rate of fraud is 9.4 %), the false negative rate is 4.11 and the false positive rate is 4.74 %. The ROC (receiving operating characteristic)for this model is 75% on the prediction of probabilities of fraud on the test set. About the hierarchy of feature importance: 
time_delta of the purchases, the country with higher fraud percentages, the source and the browser from the purchase and age are the most important variables according to the NAive Bernoulli Bayes model to understand and predict fraud probabilities. As a second approach we use support vector machines to predict fraud. This second model has a score of 0.931111307186659. We use GridSearchCV for hyper parameter tunning and we retrieve as optimal parameters {'C': 50, 'gamma': 0.005}. For these optimal parameters the score on the train set is 0.9071356992947494
againsts (a non over fitting) score on the test set of 0.9073101866149027.

* Fraud rate detected in the EDA

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/fraud_rate.jpg)

* Confusion matrix and scores using Naive Bayes model:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/fraud_confusion_matrix_naive_abyes.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/fraud_rate_naive_bayes.jpg)

* Feature importance hierarchy using Naive Bayes model:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/fraud_feature_imp_naive_bayes.jpg)


* ROC using Naive Bayes model:


![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/roc_fraud.jpg)


* Scores using SVM classifiers with GridSearch:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/sVC_score_grid_search.jpg)


## Resources and code used:

**Python version:** 3.7
**Libraries/packages used:** pandas, numpy, scikitlearn, matplotlib, seaborn
**Dataset:** available at the repo






# [Project 8. NLP MiniProject](https://github.com/Andreoliveira85/NLP-Miniproject) <a name="P8"></a>
## Project description:

In this NLP project we will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews.

Each observation in this dataset is a review of a particular business by a particular user.

The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.

The "cool" column is the number of "cool" votes this review received from other Yelp users. 

All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.

The "useful" and "funny" columns are similar to the "cool" column.

## Methodologies:

After bivariate visualization analysis we construct a ML Pipeline based on the vectorization of the text of the reviews and a Multinomial Naive Bayes model to predict probability of bad reviews (1 star) or good reviews (5 stars). The classification report is below:  
            
            
            
                precision    recall  f1-score   support

               1       0.91      0.68      0.78       228
               5       0.93      0.98      0.96       998

        accuracy                           0.93      1226
       macro avg       0.92      0.83      0.87      1226
    weighted avg       0.93      0.93      0.92      1226

## Code and resources used:

**Python version:** 3.7
**Libraries/packages:** pandas, numpy, scikitlearn.
**Dataset:** available at the repo







# [Project 9. Classification of galaxies, stars and quasars](https://github.com/Andreoliveira85/classification-galaxies-stars-and-quasars)   <a name="P9"></a>
## Project description: 

This project aims to use machine learning classification techniques to classify stellar objects: galaxies, stars and quasars. We are using data from the Sloan Digital Sky Survey. The Sloan Digital Sky Survey is a project which offers public data of space observations. 

For this purpose a special 2.5 m diameter telescope was built at the Apache Point Observatory in New Mexico, USA. The telescope uses a camera of 30 CCD-Chips with 2048x2048 image points each. The chips are ordered in 5 rows with 6 chips in each row. Each row observes the space through different optical filters (u, g, r, i, z) at wavelengths of approximately 354, 476, 628, 769, 925 nm.

The telescope covers around one quarter of the earth's sky - therefore focuses on the northern part of the sky.

**For more information about this project - please visit their website. Our dataset is provided there**

http://www.sdss.org/



## Methodologies:

We start exploring visually the data in order to try to understanding patterns in the quantitative variables gathered from the measurements. In order to solve the classification problem, after doing some feature engineering and a train/test split of 33% size, we use the ensemble learning XGBoost algorithm in order to retrieve the features hierarchy importances. In a second step in order to validate the model we use GridSearchCV with 2 folds and obtain the score on the test set  for the 1st fold 0.9904477611940299 and 0.991044776119403 for the second.

* Equatorial coordinates of the 3 class objects

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/equatorial_coords.jpg)

* Feature importance of the main explanatory variables

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/feature_importance_galaxies.jpg)

* Correlation between the main variables

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/correlations.jpg)

## Code and resources used:
**Python version:** 3.7
**Packages/libraries used:** pandas, numpy, matplotlib, seaborn, scikitlearn
**Dataset:** available at http://www.sdss.org/




# [Project 10. Miniproject DL: Breast cancer detection](https://github.com/Andreoliveira85/MiniProject-DL-Breast-cancer-detection)   <a name="P10"></a>


## Project description:

This miniproject consists in using a CNN neural net model to predict (classification) breast cancer detection on the public dataset of breast cancer available in the package datasets of scikitlearn.

## Methodologies:

 After the usual ML pipeline with a train/test split of size 20% we train a CNN neural net with a binary cross-entropy loss of  0.0496 and accuracy 0.9805 on the train set against a loss of 0.0747 and accuracy 0.9737 on the test validation set. We confront predictions with the values on the test set and visualize the plots of the loss and the accuracy metrics on the train and test sets.

* Performance of the algorithm:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/breast_cancer.jpg)

## Resources and code used:
**Python version:** 3.7
**Libraries/packages used:** pandas, numpy, tensorflow, keras, scikitlearn



# [Project 11-DL project: Yolo object detection](https://github.com/Andreoliveira85/NLP-Project-with-Deep-Learning)  <a name="P11"></a>
## Project description:
### Object Detection with YoloV3


The detection of objects in an image is one of the major application areas of Deep Learning.

The principle is simple: in addition to training an algorithm to detect and tell what is in an image, it will be trained to tell where the object is in the image:

![Object Detection](https://drive.google.com/uc?export=view&id=1G-mbb6drlUlXsMdg8Xld4p4EGYo8einf)

To do so, we will implement an algorithm called YoloV3.

However, it is very difficult to set up the whole training process of the algorithm. That's why we will learn how to use it thanks to this Github repository:

[Implement YoloV3](https://github.com/zzh8829/yolov3-tf2)

So our goal is to:

  1. Clone this repository on your local file
  2. Use it for image detection.
  3. Then try to do the same thing with video detection.
  
  ## Code/resources used:
  **Python version:** 3.7
  **Libraries/packages:** numpy, tensorflow, keras


# [Project 12. Tinder recommendation system](https://github.com/Andreoliveira85/Tinder-recommendation-system)   <a name="P12"></a>
## Company's Description 📇

<a href="https://tinder.com/?lang=en" target="_blank">Tinder</a> is of one the most famous online dating application in the world right now. The whole idea is to being able to anonymously swipe right or left on a person to show if you liked her or not. If both person swiped right: *It's a match*! And then you can start chatting! 

This whole concept revolutionized the way young people date. Founder <a href="https://www.crunchbase.com/person/sean-rad" target="_blank">Sean Rade</a> believed that *"no matter who you are, you feel more comfortable approaching somebody if you know they want you to approach them."*

With over 50 million users (80% + from 16 to 34), Tinder's valuation is around $1.4 billion which makes this start-up one of the most famous unicorn in california as of today. 😮

## Goals 🎯
Our goal is to

* Recommend 10 best profiles for a given user 

## Scope of this project 🖼️

The dataset is availabale here:


👉👉 <a href="https://full-stack-bigdata-datasets.s3.eu-west-3.amazonaws.com/Unsupervised_Learning/Tinder_data.zip" target="_blank">Tinder Data</a> 👈👈

The files contain 17,359,346 anonymous ratings of 168,791 profiles made by 135,359 LibimSeTi users. You will find a zip file contained data on gender for each person as well as a rating. 

### Introduction to Recommendation engines 

It is now time to discuss Recommendation Engines. 
There are two types of recommendation engines: 

1. Collaborative Filtering 
2. Content Based 

![](https://miro.medium.com/max/690/1*G4h4fOX6bCJhdmbsXDL0PA.png)


### Collaborative filtering principles 

We start this project with Collaborative Filtering. The idea is to recommend a product based on other users' review. Let u see the idea behind  [this explanatory gif](https://www.kdnuggets.com/2019/09/machine-learning-recommender-systems.html#:~:text=Recommender%20systems%20are%20an%20important,to%20follow%20from%20example%20code.) from KDNugget: 

![](https://miro.medium.com/max/623/1*hQAQ8s0-mHefYH83uDanGA.gif)


Instead of having "products" to recommend, this time, we will recommend people!

### Build a utility matrix 

Our goal is to be able to create a recommandation engine built on a utility matrix like this one <a href="https://towardsdatascience.com/math-for-data-science-collaborative-filtering-on-utility-matrices-e62fa9badaab" target="_blank">utility matrix</a>. This should look something like this: 

<img src="https://full-stack-assets.s3.eu-west-3.amazonaws.com/images/utility_matrix.png"/>

### Machine Learning

TruncatedSVD is the perfect algorithm here gue to the sparsity of the utility matrix! 👏 We will apply this algorithm to reduce dimension and then create a <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html" target="_blank">correlation matrix</a> to see which profile are correlated and thefore would be a match!  


## Deliverable 📬

Goals: 

* Have built a utility matrix 
* Have created a correlation matrix 
* Recommend a list of 10 profiles for a random user 



## Code/resources used:
**Python version:** 3.7
**Libraries/packages used:** pandas, numpy, scikitlearn

# [Project 13. Clustering for Uber Pickups](https://github.com/Andreoliveira85/Uber-Pickups)   <a name="P13"></a>
## Aim of the project:

One of the main pain point that Uber's team found is that sometimes drivers are not around when users need them. For example, a user might be in San Francisco's Financial District whereas Uber drivers are looking for customers in Castro.  

(check out <a href="https://www.google.com/maps/place/San+Francisco,+CA,+USA/@37.7515389,-122.4567213,13.43z/data=!4m5!3m4!1s0x80859a6d00690021:0x4a501367f076adff!8m2!3d37.7749295!4d-122.4194155" target="_blank">Google Maps</a>)

Eventhough both neighborhood are not that far away, users would still have to wait 10 to 15 minutes before being picked-up, which is too long. Uber's research shows that users accept to wait 5-7 minutes, otherwise they would cancel their ride. 

Therefore, our project aims to retrieve a recommendation system such that **their app would recommend hot-zones in major cities to be in at any given time of day.**  

### We will focus on:
* Creating an algorithm to find hot zones DBSCAN vs Kmeans 
* Visualizing results on a nice dashboard 
* We will focus on NYC city data.

## Visualizations of the clusters by Kmeans (evolution during hourly periods):

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/uber_kmeans1.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/uber_kmeans2.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/uber_kmeans3.jpg)

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/uber_kmeans4.jpg)


## Visualizations of the clusters using DBSCAN:

![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/uber_dbscan.jpg)


## Code/Resources used:
**Python version:** 3.7
**Libraries/packages used:** pandas, numpy, scikitlearn




# [Project 14. NLP-Project-with-Deep-Learning: Construction of a Translator machine](https://github.com/Andreoliveira85/NLP-Project-with-Deep-Learning) <a name="P14"></a>

## Project description:

According to the Google paper [*Attention is all you need*](https://arxiv.org/abs/1706.03762), you only need layers of Attention to make a Deep Learning model understand the complexity of a sentence. We will try to implement this type of model for our translator. 

## Project description 

 

Our data can be found on this link: https://go.aws/38ECHUB

### Preprocessing 

The whole purpose of your preprocessing is to express your (French) entry sentence in a sequence of clues.

i.e. :

* je suis malade---> `[123, 21, 34, 0, 0, 0, 0, 0]`

This gives a *shape* -> `(batch_size, max_len_of_a_sentence)`.

The clues correspond to a number that we will have to assign for each word token. 

The zeros correspond to what are called [*padded_sequences*](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) which allow all word sequences to have the same length (mandatory for your algorithm). 

This time, we won't have to *one hot encoder* your target variable. We  will simply be able to create a vector similar to your input sentence. 

i.e. : 

* I am sick ---> `[43, 2, 42, 0, 0]`

WARNING, we  will however need to add a step in our preprocessing. For each sentence we will need to add a token `<start>` & `<end>` to indicate the beginning and end of a sentence. We can do this via `Spacy`.

We will use : 

* `Pandas` or `Numpy` for reading the text file.
* `Spacy` for Tokenization 
* `Tensorflow` for [padded_sequence](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) 

### Modeling 

For modeling, we will need to set up layers of attention. We will need to: 

* Create an `Encoder` class that inherits from `tf.keras.Model`.
* Create a Bahdanau Attention Layer that will be a class that inherits `tf.keras.layers.Layer`
* Finally create a `Decoder` class that inherits from `tf.keras.Model`.


We will need to create your own cost function as well as our own training loop. 


### Tips 

We will not take the whole dataset at the beginning for our experiments, we just take 5000 or even 3000 sentences. This will allow us to iterate faster and avoid bugs simply related to your need for computing power. 

Also, we acknowledge the inspiration from the [Neural Machine Translation with Attention] tutorial (https://www.tensorflow.org/tutorials/text/nmt_with_attention) from TensorFlow. 

## Code/resources used:**
**Python version:** 3.7
**Libraries/packages used:** keras, tensorflow, numpy, pandas, spacy


# [My first data science project: Anomalies of temperatures in RJ and rainfalls of SP](https://github.com/Andreoliveira85/My-first-data-project) <a name="P15"></a>

## Project description:

### Title of the Project: Anomalies of temperatures in RJ and rainfalls of SP

### Client profile:
Our client is typically a think tank company that wants to start from the
scratch to build a decision model to economise the water reservoirs of the
state of Rio de Janeiro in the upcoming years. The reason is to try to avoid
calamity situations such as the one that happened in SP during the period
of 2014-2017 in the absence of rainfall. The critical anomaly temperature
that is taken as threshold is the value 0.62. The choice was made a prior as
a legitimate benchmark for a deviation of temperature since it was the
value registered at 2010, the year that ends the warmest decade
registered in the globe.
See https://www.ncdc.noaa.gov/sotc/global/201013.

### Questions:
1. How bad were the levels of precipitation in SP (2013-2016)?
2. How comparable were the values of the temperature anomalies in
SP in that period with historical records?
3. How do the temperatures on RJ relate and how to predict high
occurrences of temperature anomalies bigger than 0.62?
Sources:
* Temperatures (monthly 1961-2017): https://
data.giss.nasa.gov/gistemp/
* Pluviosity SP ( annual 1961-2017): https://
www.kaggle.com/arkanius/rain-intensitty-in-so-paulo/metadata
* CO2 (annual 1961-2017):
* https://www.kaggle.com/srikantsahu/co2-and-ghg-emission-data?
select=emission+data.csv

### Methodology:
1. Anomalies of temperatures: 3 data sets of different Brazilian cities
were used: SP, RJ and Manaus. The benchmark temperature for
each city was the median temperature registered monthly in each
data set. A temperature anomaly is the absolute value of the
deviations between the registered temperatures and the benchmark
temperature.

2. Import and cleaning of the data. We dealt with different data sets
and there was some previous work to create two final data frames:
one for the annual rainfall precipitation/mm^3 in SP, the annual
averaged anomalies of temperatures registered and the global levels
of emission C02 in Brazil; the second containing the anomalies of
temperatures registered in SP, RJ and Manaus on a monthly basis.
Some non negligible amount (as it is commented in the code) of
entries on the data sets were missing and several adjustments of
the data types and corresponding missing values were made during
the cleaning phase.

3. Data visualisation using matplotlib- Pyhton combined with several
graphics in Tableau. The relevant graphics are shown in the
diaporama and on the code. Exploring visually the data we draw
several conclusions: the critical values of rainfall in the period after
2000 were registered in 2014 with a tendency of high decrease
starting in 2010. At same time the anomalies of temperatures in SP
rise with an almost monotonic tendency accomplished by the
anomalies of temperatures in RJ.
4. First statistics: the annual rainfalls in SP and the emissions of C02 do
not show correlations with the annual rainfalls registered in SP. This
is a point of valid discussion such as it was made on the code file.
We used Spearman coefficient calculations since we have not a very
big data set of annual registers and we can not assume that the
underlying distributions are Normal (we can not perform the Central
Limit theorem since the observed annual values are in limited
number). Nevertheless the Spearman coefficients between the the
anomalies of temperatures are high as commented in the code
which suggested us the use of regression techniques in order to
predict models.

5. Predictions using regressions: we use multilinear regression
between the variables anomalies of temperatures for SP,RJ and
Manaus. The regression test presented high score (R^2 above 80%)
and the model predicts that the anomaly temperatures of RJ follow
the same speed as the anomaly temperatures of SP under this
model. Please see diaporama and code.

6. Predictions using the logistic regression. Under the logistic model it
is concluded with high (almost 90%) of precision and accuracy on
the data set (see the diaporama and the code file please for the
numbers and precise structure in the comments of the
implementation of this algorithm) the probability of getting a
anomaly temperature above 0.62 is modelled by a logarithmic
relation between the temperature anomalies of SP and Manaus
where the occurrence of anomalies bigger than 0.62 occur often
(check confusion matrix in the code file). This means that if in the
near future we expect time series of anomalies in the temperatures
of Rio similar ou “very close” of parts of the data set that is treated
then we can expect that under this model those anomalies are
bigger than the critical value 0.62 which was the anomaly registered
in 2010, a remarkable global warm year that closed the warmest
decade registered on the planet.

### Comments:
This deliverable is just descriptive and does not include graphics or
quantitative results that were achieved during this exploratory analysis of
the data. Climate (modelled in local geographical or global geographical
scales) is highly nonlinear and mathematically complex, not being by
nature in any way able to be captured by linear models. For the future it is
relevant to rethink in the way to associate the variables of temperature
anomalies in SP with the variables of C02 emissions and the annual
rainfalls in different ways and using different and larger data sets. The first
data frame that we used to do a first analysis is too small and also other
importante variables such as deforestation area of Amazonia, atmospheric
pressure and exogenous climatic variables such as proxy-data given by the
gulf stream should be taken into account.

### Preview of some visualizations [(link for the slides used to present the project)](https://github.com/Andreoliveira85/My-first-data-project/blob/main/andre_presentation_essentials_jedha.pdf)

![](https://github.com/Andreoliveira85/My-first-data-project/blob/main/first_project_image1.jpg)

![](https://github.com/Andreoliveira85/My-first-data-project/blob/main/first_project_image2.jpg)

![](https://github.com/Andreoliveira85/My-first-data-project/blob/main/first_project_image3.jpg)

![](https://github.com/Andreoliveira85/My-first-data-project/blob/main/first_project_image4.jpg)

![](https://github.com/Andreoliveira85/My-first-data-project/blob/main/first_project_image5.jpg)



## Code/resources used:
**Python version:** 3.7
**Packages:** pandas, numpy, scikit learn, stats, scipy
**Data vis:** Tableau


  
