import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data with caching
@st.cache_data
def load_data():
    data = pd.read_csv('European_Bank (1).csv')
    return data

def main():
    st.set_page_config(page_title="EU Bank Churn Analysis", layout="wide")
    st.title('📊 European Bank Customer Churn Analysis')
    
    # Load the data
    data = load_data()
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    total_customers = data.shape[0]
    churned_customers = data[data['Churn'] == 1].shape[0]
    churn_rate = (churned_customers / total_customers) * 100
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Age filter
    min_age = int(data['Age'].min())
    max_age = int(data['Age'].max())
    selected_age_range = st.sidebar.slider('Age Range', min_age, max_age, (25, 60))
    
    # Geography filter
    geographies = data['Geography'].unique()
    selected_geographies = st.sidebar.multiselect('Geography', geographies, default=geographies)
    
    # Apply filters
    filtered_data = data[
        (data['Age'] >= selected_age_range[0]) & 
        (data['Age'] <= selected_age_range[1]) &
        (data['Geography'].isin(selected_geographies))
    ]
    
    # Churn by Geography
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📍 Churn by Geography')
        geography_churn = filtered_data.groupby('Geography')['Churn'].agg(['sum', 'count'])
        geography_churn.columns = ['Churned', 'Total']
        geography_churn['Churn_Rate'] = (geography_churn['Churned'] / geography_churn['Total'] * 100).round(2)
        
        fig_geo = px.bar(geography_churn, x=geography_churn.index, y='Churn_Rate', 
                         title='Churn Rate by Geography', labels={'y': 'Churn Rate (%)'})
        st.plotly_chart(fig_geo, use_container_width=True)
    
    with col2:
        st.subheader('👥 Churn by Gender')
        gender_churn = filtered_data.groupby('Gender')['Churn'].agg(['sum', 'count'])
        gender_churn.columns = ['Churned', 'Total']
        gender_churn['Churn_Rate'] = (gender_churn['Churned'] / gender_churn['Total'] * 100).round(2)
        
        fig_gender = px.pie(gender_churn, values='Total', names=gender_churn.index, 
                            title='Customer Distribution by Gender')
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Age distribution
    st.subheader('📈 Age vs Churn Rate')
    age_churn = filtered_data.groupby(pd.cut(filtered_data['Age'], bins=10))['Churn'].agg(['sum', 'count'])
    age_churn.columns = ['Churned', 'Total']
    age_churn['Churn_Rate'] = (age_churn['Churned'] / age_churn['Total'] * 100).round(2)
    
    fig_age = px.line(age_churn, y='Churn_Rate', markers=True, title='Churn Rate by Age Group')
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Filtered data summary
    st.subheader('📋 Filtered Data Summary')
    st.write(f'**Customers in filtered view:** {filtered_data.shape[0]:,}')
    st.write(f'**Churned (filtered):** {filtered_data[filtered_data["Churn"] == 1].shape[0]:,}')
    churn_rate_filtered = (filtered_data[filtered_data['Churn'] == 1].shape[0] / filtered_data.shape[0]) * 100
    st.write(f'**Churn Rate (filtered):** {churn_rate_filtered:.2f}%')
    
    # Data table
    st.subheader('📊 Data Preview')
    st.dataframe(filtered_data.head(10), use_container_width=True)

if __name__ == "__main__":
    main()