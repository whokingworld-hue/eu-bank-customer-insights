import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache
def load_data():
    data = pd.read_csv('path_to_your_csv.csv')  # Replace with the actual path to the CSV file
    return data

# Main function to run the app
def main():
    st.title('European Bank Customer Churn Analysis')
    
    # Load the data
    data = load_data()
    
    # Display key metrics
    total_customers = data.shape[0]
    churned_customers = data[data['Churn'] == 1].shape[0]
    churn_rate = (churned_customers / total_customers) * 100

    st.write(f'Total Customers: {total_customers}')
    st.write(f'Churned Customers: {churned_customers}')
    st.write(f'Churn Rate: {churn_rate:.2f}%')

    # Churn by Geography
    st.subheader('Churn by Geography')
    geography_count = data.groupby(['Geography', 'Churn']).size().unstack()
    geography_count.plot(kind='bar', stacked=True)
    plt.title('Churn by Geography')
    st.pyplot(plt)

    # Interactive filter for demographics
    st.sidebar.subheader('Filters')
    selected_age_range = st.sidebar.slider('Age', int(data['Age'].min()), int(data['Age'].max()), (25, 60))
    filtered_data = data[(data['Age'] >= selected_age_range[0]) & (data['Age'] <= selected_age_range[1])]

    # Churn Rate for Selected Age Group
    st.subheader('Churn Rate for Selected Age Group')
    churn_rate_filtered = (filtered_data[filtered_data['Churn'] == 1].shape[0] / filtered_data.shape[0]) * 100
    st.write(f'Churn Rate: {churn_rate_filtered:.2f}%')

if __name__ == "__main__":
    main()