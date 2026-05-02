
import streamlit as st
import pandas as pd
import plotly.express as px

# Initialize app layout
st.set_page_config(layout="wide", page_title="European Banking Churn Analytics")
st.title("Customer Segmentation & Churn Pattern Analytics in European Banking")
st.markdown("Analyze customer churn patterns across European banking segments")

# File upload section
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload Bank Customer CSV", type="csv", icon="📂")
    
    if not uploaded_file:
        st.info("📤 Upload a CSV file to begin analysis")
        st.stop()

# Load and validate dataset
try:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.upper().str.replace(' ', '_')
    
    EXPECTED_COLS = {'CUSTOMERID', 'CREDITSCORE', 'GEOGRAPHY', 'GENDER', 'AGE', 'TENURE', 'BALANCE', 'NUMOFPRODUCTS', 'HASCRCARD', 'ISACTIVEMEMBER', 'ESTIMATEDSALARY', 'EXITED'}
    missing_cols = EXPECTED_COLS - set(df.columns)
    
    if missing_cols:
        st.warning(f"🔴 Missing columns detected: {', '.join(missing_cols)}")
        st.stop()

except Exception as e:
    st.error(f"⚠️ Data loading failed: {str(e)}")
    st.stop()

# Data preprocessing
df = df.rename(columns={
    'CUSTOMERID': 'CustomerId',
    'EXITED': 'Exited',
    'ISACTIVEMEMBER': 'IsActiveMember',
    'NUMOFPRODUCTS': 'NumOfProducts',
    'CREDITSCORE': 'CreditScore',
    'BALANCE': 'Balance',
    'ESTIMATEDSALARY': 'EstimatedSalary'
})

df['GEOGRAPHY'] = df['GEOGRAPHY'].astype('category').cat.set_categories(['France', 'Spain', 'Germany'], ordered=True)
df['Exited'] = df['Exited'].astype(int)
df['IsActiveMember'] = df['IsActiveMember'].astype(int)

# Create segmentation fields
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 30, 45, 60, 120], 
                        labels=['<30', '30-45', '46-60', '60+'])
df['CREDIT_SCORE_BAND'] = pd.cut(df['CREDITSCORE'], bins=[300, 600, 750, 850], 
                                labels=['Low', 'Medium', 'High'])
df['TENURE_GROUP'] = pd.cut(df['TENURE'], bins=[0, 2, 5, 100], 
                          labels=['New', 'Mid-term', 'Long-term'])
df['BALANCE_SEGMENT'] = pd.cut(df['BALANCE'], bins=[-1, 0, 100000, 1e7], 
                             labels=['Zero-balance', 'Low-balance', 'High-balance'])

st.sidebar.success("✅ Data loaded and processed successfully!")

# Main analytics dashboard
try:
    # KPI Section
    with st.spinner("Calculating key metrics..."):
        total_customers = len(df)
        churned_customers = df[df['Exited'] == 1].shape[0]
        churn_rate = churned_customers / total_customers * 100
        
        high_value_churn = df[df['BALANCE'] > 100000]['Exited'].mean() * 100
        active_churn = df[df['IsActiveMember'] == 1]['Exited'].mean() * 100
        inactive_churn = df[df['IsActiveMember'] == 0]['Exited'].mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", total_customers)
        col2.metric("Churned Customers", churned_customers, f"{churn_rate:.1f}% churn rate")
        col3.metric("Active Member Churn", f"{active_churn:.1f}%")
        col4.metric("High-Value Churn", f"{high_value_churn:.1f}%")
    
    # Visualization Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Geographic Analysis", "Demographic Analysis", 
                                    "Financial Segmentation", "High-Value Churn"])
    
    with tab1:
        st.header("Churn Analysis by Geography")
        col1, col2 = st.columns(2)
        
        with col1:
            geo_churn = df.groupby('GEOGRAPHY')['Exited'].value_counts().unstack()
            fig = px.bar(geo_churn, title="Churn Distribution by Country", 
                        labels={'value': 'Customers', 'GEOGRAPHY': 'Country'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            geo_churn_pct = df.groupby('GEOGRAPHY')['Exited'].mean().sort_values()
            fig = px.bar(geo_churn_pct, x=geo_churn_pct.index, y=geo_churn_pct.values,
                         title="Churn Rate by Country",
                         labels={'x': 'Country', 'y': 'Churn Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Demographic Churn Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='AGE_GROUP', y='Exited', color='GENDER',
                         title="Age Group vs Gender Churn Trends")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='AGE', y='TENURE', color='Exited',
                           hover_data=['BALANCE', 'CREDITSCORE'],
                           title="Demographic Distribution: Age vs Tenure")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Age & Tenure Cohort Analysis")
        cohort_table = pd.crosstab(df['AGE_GROUP'], df['TENURE_GROUP'], margins=True)
        fig = px.bar(cohort_table.iloc[:-1, :-1], title="Customer Cohorts by Age and Tenure",
                     labels={'value': 'Customer Count', 'AGE_GROUP': 'Age Group', 'TENURE_GROUP': 'Tenure Group'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Financial Churn Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.density_heatmap(df, x='CREDITSCORE', y='BALANCE', facet_col='Exited',
                                   title="Credit Score vs Balance Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            product_churn = df.groupby('NUMOFPRODUCTS')['Exited'].mean().reset_index()
            fig = px.line(product_churn, x='NUMOFPRODUCTS', y='Exited',
                         title="Product Holding vs Churn Risk",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Credit Risk Segmentation")
        risk_analysis = df.groupby(['CREDIT_SCORE_BAND', 'BALANCE_SEGMENT'])['Exited'].mean().unstack()
        fig = px.imshow(risk_analysis, title="Churn Risk Heatmap: Credit Score vs Balance",
                        labels=dict(x="Balance Segment", y="Credit Score", color="Churn Rate"),
                        aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("High-Value Customer Analysis")
        high_value_df = df[df['BALANCE'] > 100000]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = px.pie(high_value_df, names='GEOGRAPHY', title="High-Value Customer Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.histogram(high_value_df, x='AGE', title="Age Distribution of Premium Customers")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            hv_churn = high_value_df.groupby('AGE_GROUP')['Exited'].mean().reset_index()
            fig = px.bar(hv_churn, x='AGE_GROUP', y='Exited',
                        title="Churn Rate by Age Group (High-Value Customers)",
                        labels={'Exited': 'Churn Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("High-Value Churn Drivers")
        fig = px.scatter(high_value_df, x='ESTIMATEDSALARY', y='BALANCE', size='CREDITSCORE',
                        color='Exited', hover_name='CUSTOMERID',
                        title="Financial Profile Analysis of High-Value Customers")
        st.plotly_chart(fig, use_container_width=True)
    
    st.success("📊 Analytics complete! Explore the tabs to uncover churn patterns.")

except Exception as e:
    st.error(f"🚨 Error generating visualizations: {str(e)}")
    st.warning("See logs for details. Common fixes:")
    st.markdown("""
    1. Ensure your CSV matches the required column format
    2. Check for missing values in critical columns
    3. Verify numerical columns don't contain text values
    """)

