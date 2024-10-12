# Restaurant Analysis and Price Prediction System

## 1. Introduction
The *Restaurant Analysis and Price Prediction System* is a comprehensive web application designed to help users find restaurant recommendations, predict dining costs, and analyze sentiment based on customer reviews and ratings. Built using Python, Streamlit, and various machine learning models, this system provides an intuitive interface for both customers and restaurant owners to explore dining options and performance metrics.

## 2. Project Structure
The project is structured as follows:
- *data_recom.csv*: The dataset containing pre-processed restaurant information.
- *rf_model.pkl*: Pre-trained Random Forest model for predicting restaurant costs.
- *Streamlit Application (final.py)*: The main file that runs the application.
- *Visualizations*: Interactive plots and charts for sentiment analysis and recommendations.
- *Model Files*: Pickle files for city, locality, and cuisine label encoders.

## 3. Installation
To set up the project, follow these steps:
1. Clone the repository or download the project files.
2. Install the required Python packages by running:
   bash
   pip install -r requirements.txt
   
3. Run the Streamlit app using the following command:
   bash
   streamlit run final.py
   

## 4. Usage
Once the application is running, navigate through the sidebar to access different features:
- *Home: Suggestion*: Select a city and locality to receive restaurant recommendations.
- *Cost Prediction*: Input features like city, locality, and cuisine to predict the average cost for two.
- *Sentiment Analysis*: Visualize sentiment breakdowns and key insights based on restaurant reviews and ratings.

## 5. Features
### Restaurant Recommendation System
- Select a city and locality to get a list of top-rated restaurants.
- Restaurants are sorted by aggregate rating and votes.
- View important restaurant details such as cuisines, average cost for two, and direct links to Zomato.

### Cost Prediction System
- Predict the average cost for two people based on inputs like city, locality, cuisines, and service availability.
- A Random Forest Regressor model is used for predictions.
- The cost prediction is adjusted using a custom rounding function to make the result more realistic.

### Sentiment Analysis System
- Perform sentiment analysis on restaurants based on user reviews.
- Restaurants are classified into categories like Exceptional, Excellent, Good, Average, and Poor.
- Interactive visualizations show trends in restaurant ratings, votes, and more.

## 6. Data Preprocessing
The original dataset contained 29,753 records from various countries. However, due to insufficient data from many regions, the dataset was reduced to 8,672 records focusing on India, where the data was more comprehensive and reliable. The following preprocessing steps were applied:
- *Duplicate Removal*: Duplicates were removed to maintain data integrity.
- *Handling Missing Values*: Missing data in critical fields was either filled or removed.
- *Feature Selection*: Important features like city, locality, cuisines, ratings, and votes were retained for model building.

## 7. Models and Algorithms
### Cost Prediction Model
- *Algorithm*: A Random Forest Regressor was used to predict the average cost for two.
- *Model Hypertuning*: Hyperparameter tuning was performed using GridSearchCV to optimize parameters like n_estimators and max_depth, achieving an accuracy of 90%.

### Sentiment Analysis
- Sentiment categories were created based on aggregate ratings.
- Restaurants were classified into five categories: Exceptional, Excellent, Good, Average, and Poor.
- *RDS Integration*: Sentiment data is stored in AWS RDS for efficient querying and real-time insights.

## 8. Results and Evaluation
- The Random Forest Regressor model performed well with a *Root Mean Square Error (RMSE)* of 0.38 and a *Mean Absolute Error (MAE)* of 0.12.
- The sentiment analysis provided valuable insights into restaurant performance, allowing for effective customer sentiment tracking.

## 9. Future Enhancements
- *NLP Analysis*: Integrate natural language processing (NLP) to analyze customer reviews more deeply.
- *Global Data Integration*: Expand the dataset to include more entries from different countries.
- *Personalized Recommendations*: Implement user-specific preferences to make recommendations more personalized.
