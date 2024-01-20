# Capstone-Project_Forecasting Natural Gas Daily Consumption
This is the final project repo for the Lighthouse Lab Data Science Capstone Project, which investigate various models to predict in advance the Natural Gas daily consumption in Saskatchewan based on weather forecast.

## Problem Definition and Motivation
Natural gas is pivotal in Saskatchewan's energy landscape, particularly for heating purposes. It stands out among fossil fuels for its efficiency, producing the least CO2 per 1 MJ of energy generated. As a transition fuel, it is expected to facilitate Saskatchewan's shift towards sustainable energy. 

Natural gas consumption in Saskatchewan is subject to various influences, including weather patterns and the economic growth of heavy industries like power plants, potash mines, agri-value facilities, and refineries. Precise forecasting of natural gas usage, based on historical demands and weather conditions, is vital for ensuring a steady and reliable energy supply. For transmission companies, accurate predictions are crucial to balancing supply and demand, averting shortages, and making strategic infrastructure investments. Weather conditions, particularly extreme temperatures, significantly impact natural gas consumption. Time series models effectively capture these seasonal trends and patterns, leading to more accurate forecasts. This, in turn, enhances operational efficiency, reduces costs, and improves customer service.

### Project Objective
•	To implement various forecasting models, both classical statistical and advanced neural network-based. This includes Prophet, LightGBM, SARIMAX, LSTM, and hybrid models.

•	Focus on implementing and understanding walk-forward validation and parameter optimization. It will evaluate the forecasting performance of the different models used.

•	The project aims to demonstrate that weather factors can be used to forecast the demand for natural gas in Saskatchewan accurately. Thus contributing to the field of energy forecasting.

These objectives will guide the project towards developing an effective and accurate forecasting system for natural gas consumption in Saskatchewan.

## Project Methodology
To accomplish project objective stated above, we employed the following methodology, which includes data collection/processing, model development, model evaluation, and model forecast. See the figure below to show the process flowchart. This flowchart provides a comprehensive overview of the steps involved in predicting future natural gas demand using historical data and various machine learning models. It’s a great example of how data science can be applied to practical problems in the energy sector.

In the first first stage, the historial data on natural gas consumption was obtained from the official website of TransGas Ltd (www.transgas.com), Saskatchewan’s only natural gas transmission company, which have the exclusive right to transport natural gas within the province. While the historical weather dataset was acquired from Environment Canada opendat website for 12 weather station across Saskatchewan going back to November 1, 2013. Ten weather factors were selected, including the lowest relative humility, highest relative humility, heating degree days, cooling degree days, total precipitation, lowest temperature, average temperature, highest temperature, gust direction, and gust speed. An average of each weather factors from the 12 stations was obtined for a day to serve as an independent features for the corresponding natual gas demand.

- **Data Pre-processing and Exploratory Data Analysis** 
This involves importing and understanding the dataset, cleaning the data, detecting and handling outliers, merging the weather dataset and natural gas load, analyzing and visualizing the relationships between the different variables, time series analysis, and preparing the data for modelling. 

- **Model Development** 
The combined dataset of daily natural gas consumption and weather factors for the past 10 years is split into a learning set that spans 8.5 years (from November 1, 2013, to April 31, 2022) and a testing set that spans 1.5 years (from May 2022 to October 31, 2023). Furthermore, the learning dataset is further subdivided into training set (85%) and validation sets (15%) to train and validate the models, respectively. 
These models utilized include Prophet, LightGBM (Light Gradient Boosting Machine), SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors), LSTM (Long Short-Term Memory), Hybrid Prophet-LGBM, and Hybrid Prophet-LSTM. In addition, a baseline model whose results can be used to assess our primary ones, was slected to be a Linear Regression model because of its simplicity and efficiency. The four models are developed inside Python environment using numpy, pandas, statsmodels, Prophet, tensorflow, and Scikit-Learn. 
The Time Series Split Cross-validation method which splits data into multiple training and validation sets based on a specific time point was implemented on the learning dataset to evaluate the performance of time series models. Hyperparameter tuning was introduced in the model training phase to determined the best parameters by either grid or random search techniques, after which evaluation is done on the validation set. And the best parameters, is now used to retrained the different models, after which evaluation is done on the validation set.

- **Model Evaluation**
In the model evaluation stage, prediction performance of all the models are analyzed by comparing their test set predictions against observed test set data. The accuracy of these four models plus hybrids is compared using two statistical metrics, namely MSE, RMSE, MAE, MAPE, MDAPE, and R-squared. 

- **Predicting future Natural Gas demand**
In the model forecast stage, the model having the smallest RMSE and MAPE values is selected to forecast the natural gas demand for the next winter season in Saskatchewan.


## Conclusion
This project aimed to develop an effective and accurate forecasting system for natural gas consumption in Saskatchewan, considering various influences such as weather patterns and economic growth of heavy industries. The project successfully implemented and evaluated various forecasting models, both classical statistical and advanced neural network-based, including Prophet, LightGBM, SARIMAX, LSTM, and hybrid models.

The models were evaluated using walk-forward validation and parameter optimization, focusing on metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), Mean Directional Absolute Percentage Error (MDAPE), and R-squared.

Based on the results, the LSTM model emerged as the most accurate model, with the lowest error metrics and the highest R-squared value. This suggests that the LSTM model is capable of capturing the complex patterns in the time series data effectively, leading to more accurate forecasts.

The project demonstrated that weather factors can indeed be used to accurately forecast the demand for natural gas in Saskatchewan. This contributes to the field of energy forecasting and has practical implications for transmission companies in Saskatchewan. Accurate predictions can help these companies balance supply and demand, avert shortages, make strategic infrastructure investments, enhance operational efficiency, reduce costs, and improve customer service.

In conclusion, this project has made significant strides towards facilitating Saskatchewan's shift towards sustainable energy by improving the accuracy and efficiency of natural gas demand forecasting. However, it's important to note that model performance can vary with changes in the data, and regular retraining and validation of the models are recommended to maintain their predictive accuracy. Future work could explore other influencing factors and incorporate them into the models for even more accurate forecasts. Additionally, other advanced machine learning and deep learning models could be explored and compared. Overall, the project has laid a solid foundation for future research and development in this area.