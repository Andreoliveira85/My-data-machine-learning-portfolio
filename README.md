
# [Project 1. Sentiment-analysis-of-Tweets-and-prediction-of-bitcoin-prices](https://github.com/Andreoliveira85/Sentiment-analysis-of-Tweets-and-prediction-of-bitcoin-prices)
**Final project for concluding the Fullstack formation in Data Science at Jedha in Paris.**

## Project overview:

* Firstly we scrapped 13 institutional and private Twitter accounts using Twitter APIs from April 2018 to mid September 2020.
* We used historical datasets of Bitcoin and other top coins in the crypto space in the same time period.
* We performed an initial exploratory data analysis on the coins historical datasets in order to understand the dominance of BTC in relation to other coins and the evolution of prices in the recent years.
* We merged the twitter datasets and we runned a sentiment analysis on those approximately 60000 tweets. The main reactions to BTC were positive during that period.
* Using the index of polarity built on the tweets and the close prices of BTC in the previous day we built a LSTM model with several dense layers and a stop on the final bias. The neural network performed  well in predicting prices both on the train set (April 2018-May 2020) with a mean square error app. 2.25% and on the test set (May 2020-14 Sept 2020) with a mean square error of 3.89%. 
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
