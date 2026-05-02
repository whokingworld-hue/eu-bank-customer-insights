import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="European Banking Churn Analytics",
    layout="wide"
)

st.title("🏦 Customer Segmentation & Churn Pattern Analytics in European Banking")
st.markdown("Analyze customer churn patterns across European banking segments")

# ----------------------------
# Upload
# ----------------------------
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload Bank Customer CSV", type=["csv"])

if not uploaded_file:
    st.info("📤 Upload a CSV file to begin analysis")
    st.stop()

# ----------------------------
# Load + validate
# ----------------------------
try:
    df = pd.read_csv(uploaded_file)

    # normalize column names
    df.columns = df.columns.str.strip()

    expected_cols = {
        "CustomerId", "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "Exited"
    }

    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        st.error(f"❌ Missing columns: {', '.join(sorted(missing_cols))}")
        st.stop()

except Exception as e:
    st.error(f"⚠️ Data loading failed: {e}")
    st.stop()

# ----------------------------
# Preprocessing
# ----------------------------
# Ensure numeric types
numeric_cols = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with required numeric nulls
df = df.dropna(subset=["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "Exited"])

# Force int for binary columns
df["HasCrCard"] = df["HasCrCard"].astype(int)
df["IsActiveMember"] = df["IsActiveMember"].astype(int)
df["Exited"] = df["Exited"].astype(int)

# Geography category
df["Geography"] = df["Geography"].astype("category")

# Derived segmentation
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 45, 60, 120],
    labels=["<30", "30-45", "46-60", "60+"]
)

df["CreditScoreBand"] = pd.cut(
    df["CreditScore"],
    bins=[300, 600, 750, 900],
    labels=["Low", "Medium", "High"]
)

df["TenureGroup"] = pd.cut(
    df["Tenure"],
    bins=[-1, 2, 5, 50],
    labels=["New", "Mid-term", "Long-term"]
)

df["BalanceSegment"] = pd.cut(
    df["Balance"],
    bins=[-1, 0, 100000, 1e8],
    labels=["Zero-balance", "Low-balance", "High-balance"]
)

st.sidebar.success("✅ Data loaded and processed successfully!")

# ----------------------------
# KPI section
# ----------------------------
try:
    total_customers = len(df)
    churned_customers = (df["Exited"] == 1).sum()
    churn_rate = (churned_customers / total_customers) * 100 if total_customers else 0

    high_value_churn = df[df["Balance"] > 100000]["Exited"].mean() * 100
    active_churn = df[df["IsActiveMember"] == 1]["Exited"].mean() * 100
    inactive_churn = df[df["IsActiveMember"] == 0]["Exited"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}", f"{churn_rate:.1f}% churn rate")
    col3.metric("Active Member Churn", f"{active_churn:.1f}%")
    col4.metric("High-Value Churn", f"{high_value_churn:.1f}%")

    # ----------------------------
    # Tabs
    # ----------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Geographic Analysis",
        "Demographic Analysis",
        "Financial Segmentation",
        "High-Value Churn"
    ])

    # ---- Tab 1
    with tab1:
        st.header("Churn Analysis by Geography")
        c1, c2 = st.columns(2)

        with c1:
            geo_churn = df.groupby("Geography")["Exited"].value_counts().unstack(fill_value=0)
            fig = px.bar(
                geo_churn,
                barmode="group",
                title="Churn Distribution by Country",
                labels={"value": "Customers", "Geography": "Country"}
            )
            st.plotly_chart(fig, width="stretch")

        with c2:
            geo_churn_pct = df.groupby("Geography")["Exited"].mean().sort_values() * 100
            fig = px.bar(
                x=geo_churn_pct.index,
                y=geo_churn_pct.values,
                title="Churn Rate by Country",
                labels={"x": "Country", "y": "Churn Rate (%)"}
            )
            st.plotly_chart(fig, width="stretch")

    # ---- Tab 2
    with tab2:
        st.header("Demographic Churn Analysis")
        c1, c2 = st.columns(2)

        with c1:
            fig = px.box(
                df,
                x="AgeGroup",
                y="Exited",
                color="Gender",
                title="Age Group vs Gender Churn Trends"
            )
            st.plotly_chart(fig, width="stretch")

        with c2:
            fig = px.scatter(
                df,
                x="Age",
                y="Tenure",
                color="Exited",
                hover_data=["Balance", "CreditScore"],
                title="Demographic Distribution: Age vs Tenure"
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Age & Tenure Cohort Analysis")
        cohort_table = pd.crosstab(df["AgeGroup"], df["TenureGroup"])
        cohort_long = cohort_table.reset_index().melt(
            id_vars="AgeGroup",
            var_name="TenureGroup",
            value_name="CustomerCount"
        )
        fig = px.bar(
            cohort_long,
            x="AgeGroup",
            y="CustomerCount",
            color="TenureGroup",
            barmode="group",
            title="Customer Cohorts by Age and Tenure"
        )
        st.plotly_chart(fig, width="stretch")

    # ---- Tab 3
    with tab3:
        st.header("Financial Churn Patterns")
        c1, c2 = st.columns(2)

        with c1:
            fig = px.density_heatmap(
                df,
                x="CreditScore",
                y="Balance",
                facet_col="Exited",
                title="Credit Score vs Balance Distribution"
            )
            st.plotly_chart(fig, width="stretch")

        with c2:
            product_churn = df.groupby("NumOfProducts")["Exited"].mean().reset_index()
            product_churn["Exited"] = product_churn["Exited"] * 100
            fig = px.line(
                product_churn,
                x="NumOfProducts",
                y="Exited",
                title="Product Holding vs Churn Risk",
                markers=True,
                labels={"Exited": "Churn Rate (%)", "NumOfProducts": "Number of Products"}
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Credit Risk Segmentation")
        risk_analysis = df.groupby(["CreditScoreBand", "BalanceSegment"])["Exited"].mean().unstack()
        fig = px.imshow(
            risk_analysis,
            title="Churn Risk Heatmap: Credit Score vs Balance",
            labels=dict(x="Balance Segment", y="Credit Score Band", color="Churn Rate"),
            aspect="auto"
        )
        st.plotly_chart(fig, width="stretch")

    # ---- Tab 4
    with tab4:
        st.header("High-Value Customer Analysis")
        high_value_df = df[df["Balance"] > 100000].copy()

        c1, c2 = st.columns([1, 2])

        with c1:
            fig = px.pie(
                high_value_df,
                names="Geography",
                title="High-Value Customer Distribution"
            )
            st.plotly_chart(fig, width="stretch")

            fig = px.histogram(
                high_value_df,
                x="Age",
                title="Age Distribution of Premium Customers"
            )
            st.plotly_chart(fig, width="stretch")

        with c2:
            hv_churn = high_value_df.groupby("AgeGroup")["Exited"].mean().reset_index()
            hv_churn["Exited"] = hv_churn["Exited"] * 100
            fig = px.bar(
                hv_churn,
                x="AgeGroup",
                y="Exited",
                title="Churn Rate by Age Group (High-Value Customers)",
                labels={"Exited": "Churn Rate (%)"}
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("High-Value Churn Drivers")
        fig = px.scatter(
            high_value_df,
            x="EstimatedSalary",
            y="Balance",
            size="CreditScore",
            color="Exited",
            hover_name="CustomerId",
            title="Financial Profile Analysis of High-Value Customers"
        )
        st.plotly_chart(fig, width="stretch")

    st.success("📊 Analytics complete! Explore the tabs to uncover churn patterns.")

except Exception as e:
    st.error(f"🚨 Error generating visualizations: {e}")
    st.warning("Common fixes:")
    st.markdown("""
1. Ensure your CSV matches required columns  
2. Ensure numeric columns are valid numbers  
3. Ensure there are no empty required fields
""")
