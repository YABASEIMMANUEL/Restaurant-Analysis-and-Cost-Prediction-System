import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load dataset for both recommendation and sentiment analysis
def load_data():
    data = pd.read_csv('filtered_rs_currency_dataset.csv')  # Assuming all features are in the same dataset
    return data

data = load_data()

# Set page layout and title
st.set_page_config(
    page_title="Restaurant Analysis System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home: Suggestion", "Cost Prediction", "Sentiment Analysis"])

# Home Page: sug1final.py - Restaurant Suggestion System
if page == "Home: Suggestion":
    st.title("Restaurant Recommendation System")

    cities = data['city'].unique()
    selected_city = st.selectbox("Select City", cities, index=None)

    localities = data[data['city'] == selected_city]['locality'].unique()
    selected_locality = st.selectbox("Select Locality", localities, index=None)

    # Automatically show recommendations based on selected city and locality
    if selected_locality:
        # Filter data for selected city and locality
        recommendations = data[(data['city'] == selected_city) & (data['locality'] == selected_locality)]
        recommendations = recommendations.sort_values(by=['aggregate_rating', 'votes'], ascending=[False, False])
        
        st.subheader(f"Top Restaurants in {selected_locality}, {selected_city}")
        
        # Display recommendations in columns (two restaurants per row)
        for i in range(0, len(recommendations), 2):
            cols = st.columns(2)  # Create two columns
            
            # First restaurant (left column)
            with cols[0]:
                if i < len(recommendations):
                    row = recommendations.iloc[i]
                    st.subheader(row['name'])
                    
                    # Check if the thumbnail URL exists
                    if pd.notna(row['thumb']):
                        st.image(row['thumb'], width=200)  # Display thumbnail image
                    else:
                        st.image('https://via.placeholder.com/200', width=200)  # Display a placeholder image
                    
                    st.write(f"**Cuisines**: {row['cuisines']}")
                    st.write(f"**Cost for Two**: {row['formatted_cost']}")
                    st.write(f"**Rating**: {row['aggregate_rating']} | **Votes**: {row['votes']}")
                    st.write(f"[View on Zomato]({row['url']})")
            
            # Second restaurant (right column)
            with cols[1]:
                if i + 1 < len(recommendations):  # Check if there's a second restaurant
                    row = recommendations.iloc[i + 1]
                    st.subheader(row['name'])
                    
                    # Check if the thumbnail URL exists
                    if pd.notna(row['thumb']):
                        st.image(row['thumb'], width=200)  # Display thumbnail image
                    else:
                        st.image('https://via.placeholder.com/200', width=200)  # Display a placeholder image
                    
                    st.write(f"**Cuisines**: {row['cuisines']}")
                    st.write(f"**Cost for Two**: {row['formatted_cost']}")
                    st.write(f"**Rating**: {row['aggregate_rating']} | **Votes**: {row['votes']}")
                    st.write(f"[View on Zomato]({row['url']})")
                    
            st.write("---")  # Divider between rows

# Cost Prediction Page: te11.py - Predict Average Cost for Two
elif page == "Cost Prediction":
    st.title("Predict Average Cost for Two")

    # Load the trained Random Forest model and label encoders
    model = pickle.load(open("RandomForestRegressor.pkl", "rb"))
    le_city = pickle.load(open("le_city.pkl", "rb"))
    le_locality = pickle.load(open("le_locality.pkl", "rb"))
    le_cuisines = pickle.load(open("le_cuisines.pkl", "rb"))

    st.header("Input the following details:")

    # City selection
    city = st.selectbox("Select City", le_city.classes_)

    # Locality selection based on the city
    localities = data[data['city'] == city]['locality'].unique()
    locality = st.selectbox("Select Locality", localities)

    # Cuisines selection
    cuisines = st.selectbox("Select Cuisines", le_cuisines.classes_)

    # Online delivery selection
    has_online_delivery = st.radio("Has Online Delivery?", ("Yes", "No"))
    has_online_delivery = 1 if has_online_delivery == "Yes" else 0

    # Table booking selection
    has_table_booking = st.radio("Has Table Booking?", ("Yes", "No"))
    has_table_booking = 1 if has_table_booking == "Yes" else 0

    # Encode the inputs using label encoders
    city_encoded = le_city.transform([city])[0]
    locality_encoded = le_locality.transform([locality])[0]
    cuisines_encoded = le_cuisines.transform([cuisines])[0]

    # Prepare the data for prediction
    input_data = np.array([[has_online_delivery, has_table_booking, city_encoded, locality_encoded, cuisines_encoded]])

    # Prediction button
    if st.button("Predict Average Cost for Two"):
        predicted_cost_log = model.predict(input_data)
        predicted_cost = np.expm1(predicted_cost_log)[0]  # Inverse log transformation

        # Apply custom rounding logic
        integer_part = int(predicted_cost // 100) * 100
        decimal_part = predicted_cost % 100

        if decimal_part < 50:
            predicted_cost = integer_part  # Round down
        else:
            predicted_cost = integer_part + 100  # Round up

        st.success(f"The predicted average cost for two is ₹{predicted_cost:.0f}")

# Sentiment Analysis Page: senti2.py - Restaurant Sentiment Analysis
elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis of Restaurants")

    # Function to filter data based on city and locality
    def filter_data(city, locality):
        if city != 'All':
            data_filtered = data[data['city'] == city]
            if locality != 'All':
                data_filtered = data_filtered[data_filtered['locality'] == locality]
        else:
            data_filtered = data
        return data_filtered

    # Function to generate restaurant summary
    def generate_restaurant_summary(filtered_data):
        restaurant_summary = filtered_data.groupby('name').apply(
            lambda x: {
                "Rating": x['rating_text'].values[0],
                "Votes": x['votes'].values[0],
                "Aggregate Rating": x['aggregate_rating'].values[0],
                "Description": f"{x['rating_text'].values[0]} restaurant with {x['votes'].values[0]} votes and an aggregate rating of {x['aggregate_rating'].values[0]}"
            }
        ).reset_index()

        summary_data = pd.DataFrame(restaurant_summary[0].tolist(), index=restaurant_summary['name'])
        return summary_data

    # Customized sentiment categories function
    def categorize_sentiment(row):
        if row['aggregate_rating'] >= 4.6:
            return "Exceptional"
        elif 4.0 <= row['aggregate_rating'] < 4.6:
            return "Excellent"
        elif 3.5 <= row['aggregate_rating'] < 4.0:
            return "Good"
        elif 3.0 <= row['aggregate_rating'] < 3.5:
            return "Average"
        else:
            return "Poor"

    # Add sentiment category
    data['Sentiment'] = data.apply(categorize_sentiment, axis=1)
    # Mapping for has_online_delivery and has_table_booking
    data['online_delivery_status'] = data['has_online_delivery'].apply(lambda x: "Online delivery available" if x == 1 else "No online delivery available")
    data['table_booking_status'] = data['has_table_booking'].apply(lambda x: "Table booking available" if x == 1 else "No table booking available")
    # Filters
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox('Select City', ['All'] + sorted(data['city'].unique().tolist()))

    with col2:
        if city != 'All':
            locality = st.selectbox('Select Locality', ['All'] + sorted(data[data['city'] == city]['locality'].unique().tolist()))
        else:
            locality = 'All'

    # Filter data based on city and locality
    filtered_data = filter_data(city, locality)

    # Case 1: Default (City and Locality are both 'All')
    if city == 'All' and locality == 'All':
        st.write(f"### Showing data for all cities and localities")

        # Display all graphs when both city and locality are 'All'
        st.write("### Graphical Analysis")
        col1, col2 = st.columns(2)
        
        # 1. Distribution of Aggregate Ratings using Seaborn (left column)
        with col1:
            st.write("#### 1. Distribution of Aggregate Ratings")
            fig1, ax1 = plt.subplots(figsize=(6, 4))  # Set fixed size for consistency
            sns.histplot(filtered_data['aggregate_rating'], kde=True, bins=20, color='blue', ax=ax1)
            st.pyplot(fig1)
            st.write("")
            st.write("")
            st.write("")
            st.write("")  
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")  
        # 2. Votes vs Aggregate Rating using Plotly (right column)
        with col2:
            st.write("#### 2. Votes vs Aggregate Rating")
            fig2 = px.scatter(filtered_data, x='votes', y='aggregate_rating', color='aggregate_rating', size='votes',
                            title="Votes vs Aggregate Rating", height=400, width=600)
            st.plotly_chart(fig2)
            st.write("")
            st.write("")
            st.write("")
            st.write("")  
            st.write("")
            st.write("")

        st.write("")  # Add space
        st.write("")
        st.write("")
        # 3. Average Cost for Two vs Aggregate Rating (left column)
        with col1:
            st.write("#### 3. Average Cost for Two vs Aggregate Rating")
            fig3, ax3 = plt.subplots(figsize=(6, 4))  # Set fixed size for consistency
            sns.scatterplot(x='average_cost_for_two', y='aggregate_rating', hue='aggregate_rating', data=filtered_data, ax=ax3)
            st.pyplot(fig3)
            st.write("")
            st.write("")
            st.write("")
            st.write("")  
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("") 
        # 4. Top 10 Restaurants by Votes (right column)
        with col2:
            st.write("#### 4. Top 10 Restaurants by Votes")
            top_10_votes = filtered_data.nlargest(10, 'votes')
            fig4 = px.bar(top_10_votes, x='votes', y='name', title="Top 10 Restaurants by Votes", orientation='h', height=400, width=600)
            st.plotly_chart(fig4)
            st.write("")
            st.write("")
            st.write("")
            st.write("")  
            st.write("")
            st.write("")
            st.write("")
            st.write("")            

        st.write("")  # Add space
        st.write("")
        st.write("")
        # 5. Cities with the Most Highly Rated Restaurants (left column)
        with col1:
            st.write("#### 5. Cities with the Most Highly Rated Restaurants")
            high_rated_cities = data[data['aggregate_rating'] >= 4.6].groupby('city').size().reset_index(name='high_rating_count')
            top_high_rated_cities = high_rated_cities.sort_values(by='high_rating_count', ascending=False).head(10)
            fig5 = px.bar(top_high_rated_cities, x='city', y='high_rating_count', title="Cities with Most Highly Rated Restaurants", height=400, width=600)
            st.plotly_chart(fig5)

        # 6. Cities with the Most Low Rated Restaurants (right column)
        with col2:
            st.write("#### 6. Cities with the Most Low Rated Restaurants")
            low_rated_cities = data[data['aggregate_rating'] < 3.0].groupby('city').size().reset_index(name='low_rating_count')
            top_low_rated_cities = low_rated_cities.sort_values(by='low_rating_count', ascending=False).head(10)
            fig6 = px.bar(top_low_rated_cities, x='city', y='low_rating_count', title="Cities with Most Low Rated Restaurants", height=400, width=600)
            st.plotly_chart(fig6)

        st.write("")  # Add space
        st.write("")
        st.write("")
        # 7. Restaurants with Votes Between 0 and 3 (left column)
        with col1:
            st.write("#### 7. Restaurants with Votes Between 0 and 3")
            votes_filtered = data[(data['votes'] >= 0) & (data['votes'] <= 3)]
            votes_count = votes_filtered.groupby('votes')['name'].count().reset_index()
            votes_count.columns = ['Votes', 'Restaurant Count']
            fig7 = px.bar(votes_count, x='Votes', y='Restaurant Count', title="Restaurants with Votes Between 0 to 3",
                        labels={"Votes": "Votes", "Restaurant Count": "Number of Restaurants"})
            st.plotly_chart(fig7)

        st.write("")  # Add space
        st.write("")
        st.write("")
        # 8. Top 10 Cities by Rating Count (right column)
        with col2:
            st.write("#### 8. Top 10 Cities by Rating Count")
            city_rating_counts = data.groupby('city')['aggregate_rating'].count().reset_index()
            city_rating_counts.columns = ['city', 'rating_count']
            top_10_cities = city_rating_counts.sort_values(by='rating_count', ascending=False).head(10)
            fig8 = go.Figure([go.Bar(x=top_10_cities['city'], y=top_10_cities['rating_count'], marker_color='indigo')])
            fig8.update_layout(title="Top 10 Cities with Most Ratings", xaxis_title="City", yaxis_title="Rating Count")
            st.plotly_chart(fig8)

    # Top 5 Restaurants with Rating 4.0
        st.write("### Top 5 Restaurants with Rating 4.0 by City")
        rating_4_restaurants = data[data['aggregate_rating'] == 4.0]
        top_5_restaurants = rating_4_restaurants.groupby('city').apply(lambda x: x.nlargest(5, 'average_cost_for_two')).reset_index(drop=True)
        st.dataframe(top_5_restaurants[['name', 'city', 'average_cost_for_two', 'aggregate_rating']])


    # Case 2: City Selected, but Locality is 'All'
    elif city != 'All' and locality == 'All':
        st.write(f"### Showing locality details for city: {city}")
        st.dataframe(data[data['city'] == city][['locality', 'name', 'aggregate_rating']].drop_duplicates())

    # Case 3: Both City and Locality Selected
    elif city != 'All' and locality != 'All':
        st.write(f"### Showing restaurant summary for {locality} in {city}")
        summary_data = generate_restaurant_summary(filtered_data)
        st.dataframe(summary_data)

        restaurant_name = st.selectbox('Select Restaurant', ['None'] + sorted(filtered_data['name'].unique().tolist()))

        if restaurant_name != 'None':
            restaurant_details = filtered_data[filtered_data['name'] == restaurant_name]
            st.write(restaurant_details[['name', 'locality', 'cuisines', 'average_cost_for_two', 'aggregate_rating', 'rating_text', 'votes', 'online_delivery_status', 'table_booking_status']])
