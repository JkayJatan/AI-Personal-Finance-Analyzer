import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Finance Analyzer", page_icon="💰")
def categorize_expenses(df):
    categories = {
        'Food': ['zomato', 'swiggy', 'restaurant', 'cafe', 'coffee', 'pizza'],
        'Groceries': ['big basket', 'blinkit', 'grocery', 'supermarket'],
        'Transport': ['uber', 'ola', 'petrol', 'fuel', 'metro'],
        'Shopping': ['amazon', 'flipkart', 'myntra', 'clothes'],
        'Bills': ['electricity', 'water', 'internet', 'rent'],
        'Entertainment': ['netflix', 'spotify', 'movie', 'cinema']
    }
    if 'Description' not in df.columns:
        return df
    def get_category(desc):
        desc_lower = str(desc).lower()
        for cat, keywords in categories.items():
            if any(word in desc_lower for word in keywords):
                return cat
        return 'Other'
    df['Category'] = df['Description'].apply(get_category)
    return df

def generate_sample_csv():
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Description': ['Rent', 'Swiggy', 'Uber', 'Amazon', 'Zomato', 'Electricity', 'Grocery', 'Netflix', 'Petrol', 'Grocery', 'Restaurant', 'Internet'],
        'Amount': [-15000, -450, -200, -1500, -300, -1200, -3000, -500, -600, -2500, -800, -1000]
    }
    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')

def train_model(df):
    df['Date'] = pd.to_datetime(df['Date'])
    monthly = df.resample('M', on='Date')['Amount'].sum().reset_index()
    monthly['Month_Num'] = range(len(monthly))
    X = monthly[['Month_Num']]
    y = monthly['Amount']
    model = LinearRegression()
    model.fit(X, y)
    future = np.array([[len(monthly)], [len(monthly)+1], [len(monthly)+2]])
    return model.predict(future)

st.title("💰 AI Finance Analyzer")

with st.sidebar:
    st.header("Upload CSV")
    uploaded_file = st.file_uploader("Date, Description, Amount columns", type=['csv'])
    st.download_button("Sample CSV", generate_sample_csv(), "sample.csv", "text/csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if not all(c in df.columns for c in ['Date','Description','Amount']):
            st.error("Need: Date, Description, Amount columns")
            st.stop()
        
        df['Date'] = pd.to_datetime(df['Date'])
        expenses = df[df['Amount'] < 0].copy()
        expenses['Amount'] = expenses['Amount'].abs()
        expenses = categorize_expenses(expenses)

        total = expenses['Amount'].sum()
        avg = expenses.groupby(expenses['Date'].dt.to_period('M'))['Amount'].sum().mean()
        top = expenses.groupby('Category')['Amount'].sum().idxmax()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", f"${total:,.0f}")
        c2.metric("Monthly Avg", f"${avg:,.0f}")
        c3.metric("Top Category", top)

       
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔮 Prediction", "💡 Tips"])
        
        with tab1:
            fig = px.pie(expenses, values='Amount', names='Category', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            preds = train_model(df)
            st.write("Next 3 months:", preds)
            st.line_chart(pd.DataFrame(preds, columns=["Predicted"]))
        
        with tab3:
            st.info("Tip: Monitor your Food spending!")
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV to start")