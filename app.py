import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Earnings Call NLP Analyzer",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv('data/sentiment_results.csv')

st.title("📈 Earnings Call NLP Analyzer")
st.markdown("**FinBERT sentiment analysis on 514 earnings call transcripts across 30 major US companies**")

df = load_data()

# KPI cards
st.subheader("Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transcripts Analyzed", f"{len(df):,}")
col2.metric("Companies", f"{df['ticker'].nunique()}")
col3.metric("Quarters Covered", f"{df['quarter'].nunique()}")
col4.metric("Avg Sentiment Score", f"{df['sentiment_score'].mean():.3f}")

st.divider()

# Sidebar filters
st.sidebar.header("Filters")
sectors = st.sidebar.multiselect(
    "Sector", options=sorted(df['sector'].dropna().unique()),
    default=sorted(df['sector'].dropna().unique())
)
tickers = st.sidebar.multiselect(
    "Company", options=sorted(df['ticker'].unique()),
    default=sorted(df['ticker'].unique())
)

df_filtered = df[df['sector'].isin(sectors) & df['ticker'].isin(tickers)]

st.subheader("Sentiment by Company and Sector")
col1, col2 = st.columns(2)

with col1:
    company_sent = df_filtered.groupby('ticker')['sentiment_score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(company_sent.index, company_sent.values,
            color=['steelblue' if x > 0 else 'crimson' for x in company_sent.values])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Avg Sentiment Score by Company', fontweight='bold')
    ax.set_xlabel('Sentiment Score (Positive - Negative)')
    st.pyplot(fig)
    plt.close()

with col2:
    sector_sent = df_filtered.groupby('sector')['sentiment_score'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.bar(sector_sent.index, sector_sent.values,
           color=sns.color_palette('RdYlGn', len(sector_sent)))
    ax.set_title('Avg Sentiment Score by Sector', fontweight='bold')
    ax.set_ylabel('Sentiment Score')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close()

st.divider()

st.subheader("Sentiment Trend Over Time")
selected_tickers = st.multiselect(
    "Select companies to compare",
    options=sorted(df['ticker'].unique()),
    default=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
)

if selected_tickers:
    df_trend = df[df['ticker'].isin(selected_tickers)].sort_values('quarter')
    fig, ax = plt.subplots(figsize=(14, 5))
    for ticker in selected_tickers:
        data = df_trend[df_trend['ticker']==ticker]
        ax.plot(data['quarter'], data['sentiment_score'],
                marker='o', label=ticker, linewidth=2)
    ax.set_title('Sentiment Score Trend by Quarter', fontweight='bold')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Sentiment Score')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    st.pyplot(fig)
    plt.close()

st.divider()

st.subheader("Risk Language Analysis")
col1, col2 = st.columns(2)

with col1:
    risk_by_co = df_filtered.groupby('ticker')['risk_ratio'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(risk_by_co.index, risk_by_co.values,
            color=sns.color_palette('Reds_r', len(risk_by_co)))
    ax.set_title('Risk Language Ratio by Company', fontweight='bold')
    ax.set_xlabel('Risk Ratio')
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(df_filtered['risk_ratio'], df_filtered['sentiment_score'],
               alpha=0.6, c=df_filtered['sentiment_score'], cmap='RdYlGn', s=40)
    ax.set_title('Sentiment vs Risk Language', fontweight='bold')
    ax.set_xlabel('Risk Ratio')
    ax.set_ylabel('Sentiment Score')
    st.pyplot(fig)
    plt.close()

st.divider()

st.subheader("Data Explorer")
st.dataframe(
    df_filtered[['ticker','sector','quarter','sentiment_score',
                 'positive','negative','risk_ratio']]
    .sort_values('sentiment_score', ascending=False),
    use_container_width=True
)

st.caption("Model: FinBERT (ProsusAI) | Data: Motley Fool Earnings Transcripts | Built by Sumaksharika")
