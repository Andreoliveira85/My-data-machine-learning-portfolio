# Table of contents
- [Project 1. Sentiment-analysis-of-Tweets-and-prediction-of-bitcoin-prices](#P1)
- [Project 2. NYSE Prediction of Prices](#P2)
- [Project 3. Electronic Signature of Loans] (#P3)
- [Project 4. Hybrid Mutual Fund Analysis] (#P4)
- [Project 5. Linear Regression in E-commerce] (#P5)
- [Project 6. Advertisement with ML] (#P6)
- [Project 7. Detecting fraud with ML] (#P7)
- [Project 8. NLP MiniProject] (#P8)
- [Project 9. Classification of galaxies, stars and quasars] (#P9)
- [Project 10. Miniproject DL: Breast cancer detection] (#P10)
- [Project 11.DL project; Yolo object detection] (#P11)
- [Project 12. Tinder recommendation system] (#P12)
- [Project 13. Clustering for Uber Pickups] (#P13)
  
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
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/sentiment_analysis_tweets2.jpg)

* Split between train and test set
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/training_test_dataset.jpg)

* The results of the LSTM model on the train set (April 2018-April 2020 approx.). The model learned quite well.
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/model_training_set.jpg). 

* The results of our model on the test set (April 2020- mid Sept 2020). The trends are followed by the predictions. There is a time period where the model performs far worst than the actual prices (August-September) but then the actual prices and the predictions start to converge again in the last two weeks of the dataset almost coallescing in the end.
![https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/model_test_dataset.jpg]

* The loss function (Mean squared error) on train and test set (per epoch). Scale 10^-3
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/loss_function.jpg)

* The predictions built by our model were used as regressors to feed the FB Prophet algorithm. The seasonality effects of the predictions based on the LSTM predictors are shown below. Monthly and bigger size trends should be neglected since the coin is not mature enough for such extrapolations. Although interesting to note the weekly trend of the price going down on Thusrdays and going up during the weekends.
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/prophet_model2.jpg)

* The results of the Prophetization of our predictions:
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/prophet_model.jpg)

* The contrast of the predictions of the FB Prophet model feeded by the outputs of the LSTM net we built with the actual prices from 15 Sept 2020 until 30 November 2020 (fetched on Google Finance)
![](https://github.com/Andreoliveira85/My-data-machine-learning-portfolio/blob/main/images_folder/prophets_results_reality.jpg)

**Conclusion:** the predictions of our LSTM model when are used as predictors in the FB Prophet algorithm produce new predictions that are close of the close actual value of the BTC according to Google Finance during a couple days (15 Sept until beg October). After that the actual value takes off much more than the predictions. This can be empirically understood due to the big component of noise in this time series and the recent bubble effect registered with BTC in the crypto space that is steeper than the high derivative of prices in 2017. According to this model this would be a good time to sell.


# [Project 2. NYSE Prediction of Prices]  <a name="P2"></a>
# [Project 3. Electronic Signature of Loans]  <a name="P3"></a>
# [Project 4. Hybrid Mutual Fund Analysis]   <a name="P4"></a>
# [Project 5. Linear Regression in E-commerce]   <a name="P5"></a>
# [Project 6. Advertisement with ML]   <a name="P6"></a>
# [Project 7. Detecting fraud with ML]  <a name="P7"></a>
# [Project 8. NLP MiniProject] <a name="P8"></a>
# [Project 9. Classification of galaxies, stars and quasars]   <a name="P9"></a>
# [Project 10. Miniproject DL: Breast cancer detection]   <a name="P10"></a>
# [Project 11.DL project; Yolo object detection]  <a name="P11"></a>
# [Project 12. Tinder recommendation system]   <a name="P12"></a>
# [Project 13. Clustering for Uber Pickups]   <a name="P13"></a>
  
