import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import spacy
from textblob import TextBlob
import ast

# Ensure spaCy model is downloaded
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Load datasets
@st.cache_data
def load_data():
    dataset1 = pd.read_csv('/Users/gautham/Documents/USI/Sem 2/Text Analysis and Spatial Data for Economists/Project/IEEE DS/paper_details_final_data.csv')
    dataset2 = pd.read_csv('/Users/gautham/Documents/USI/Sem 2/Text Analysis and Spatial Data for Economists/Project/IEEE DS/database.csv')
    return dataset1, dataset2

dataset1, dataset2 = load_data()

# Print columns to debug
st.write("Dataset 1 columns:", dataset1.columns)
st.write("Dataset 2 columns:", dataset2.columns)

# Extract top 10 keywords
@st.cache_data
def extract_top_keywords(dataset):
    all_keywords = []
    for keywords in dataset['final_keywords']:
        if pd.notna(keywords):
            try:
                keyword_list = ast.literal_eval(keywords)
                if isinstance(keyword_list, list):
                    cleaned_keywords = [keyword.strip().lower() for keyword in keyword_list]
                    all_keywords.extend(cleaned_keywords)
            except:
                pass
    keyword_counts = Counter(all_keywords)
    top_keywords = [keyword for keyword, count in keyword_counts.most_common(10)]
    return top_keywords

top_keywords = extract_top_keywords(dataset1)

# Filter papers by top keywords
def contains_top_keywords(abstract, keywords):
    if pd.isna(abstract):
        return False
    doc = nlp(abstract.lower())
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return any(keyword in tokens for keyword in keywords)

filtered_papers = dataset2[dataset2['abstract'].apply(lambda x: contains_top_keywords(x, top_keywords))]

# Preprocess abstracts
filtered_papers['processed_abstract'] = filtered_papers['abstract'].apply(lambda x: ' '.join([token.text for token in nlp(x.lower()) if not token.is_stop and not token.is_punct]) if pd.notna(x) else '')

# Define the function to plot keyword trends
def plot_keyword_trends(keyword):
    keyword_trend = filtered_papers[filtered_papers['abstract'].str.contains(keyword, case=False, na=False)]
    trend_data = keyword_trend.groupby('year').size().reset_index(name='count')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=trend_data, x='year', y='count', marker='o')
    plt.title(f'Temporal Trend of Keyword: {keyword}')
    plt.xlabel('Year')
    plt.ylabel('Number of Papers')
    plt.grid(True)
    st.pyplot(plt)

# Define the function to plot top 3 keywords for a specific year
def plot_top_keywords_by_year(year):
    yearly_data = filtered_papers[filtered_papers['year'] == year]
    yearly_keywords = []

    for abstract in yearly_data['abstract']:
        doc = nlp(abstract.lower())
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        yearly_keywords.extend([keyword for keyword in tokens if keyword in top_keywords])

    keyword_counts = Counter(yearly_keywords)
    top_3_keywords = keyword_counts.most_common(3)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=[kw for kw, _ in top_3_keywords], y=[count for _, count in top_3_keywords])
    plt.title(f'Top 3 Keywords in {year}')
    plt.xlabel('Keywords')
    plt.ylabel('Count')
    plt.grid(True)
    st.pyplot(plt)

# Define the function to plot keyword distribution across different research fields
def plot_keyword_distribution_by_field(year):
    yearly_data = filtered_papers[filtered_papers['year'] == year]
    field_keywords = {}

    for _, row in yearly_data.iterrows():
        doc = nlp(row['abstract'].lower())
        tokens = [token.text for token in doc if not token is_stop and not token is_punct]
        for keyword in top_keywords:
            if keyword in tokens:
                if keyword not in field_keywords:
                    field_keywords[keyword] = Counter()
                field_keywords[keyword].update([row['year']])  # Assuming 'field' column exists

    field_data = pd.DataFrame(field_keywords).fillna(0).astype(int)
    field_data.plot(kind='bar', stacked=True, figsize=(15, 7))
    plt.title(f'Keyword Distribution Across Different Research Fields in {year}')
    plt.xlabel('Research Fields')
    plt.ylabel('Count')
    plt.legend(title='Keywords', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# Define the function to plot keyword co-occurrence
def plot_keyword_cooccurrence():
    cooccurrence_matrix = np.zeros((len(top_keywords), len(top_keywords)))

    for abstract in filtered_papers['abstract']:
        doc = nlp(abstract.lower())
        tokens = [token.text for token in doc if not token.is_stop and not token is_punct]
        keywords_in_abstract = [keyword for keyword in tokens if keyword in top_keywords]

        for i, keyword1 in enumerate(top_keywords):
            for j, keyword2 in enumerate(top_keywords):
                if keyword1 in keywords_in_abstract and keyword2 in keywords_in_abstract:
                    cooccurrence_matrix[i, j] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_matrix, xticklabels=top_keywords, yticklabels=top_keywords, cmap='Blues', annot=True)
    plt.title('Keyword Co-occurrence Matrix')
    plt.xlabel('Keywords')
    plt.ylabel('Keywords')
    plt.grid(False)
    st.pyplot(plt)

# Streamlit app layout
st.title('Research Paper Analysis')
st.sidebar.title('Select Analysis')

# Dropdown menu for selecting the analysis
analysis = st.sidebar.selectbox('Analysis', ['Keyword Trends Over Time', 'Top Keywords by Year', 'Keyword Distribution by Field', 'Keyword Co-occurrence'])

if analysis == 'Keyword Trends Over Time':
    # Create a dropdown menu with the top 10 keywords
    keyword = st.sidebar.selectbox('Keyword', top_keywords)
    if keyword:
        st.subheader(f'Temporal Trend of Keyword: {keyword}')
        plot_keyword_trends(keyword)

elif analysis == 'Top Keywords by Year':
    # Create a dropdown menu with the years
    year = st.sidebar.selectbox('Year', filtered_papers['year'].unique())
    if year:
        st.subheader(f'Top 3 Keywords in {year}')
        plot_top_keywords_by_year(year)

elif analysis == 'Keyword Distribution by Field':
    # Create a dropdown menu with the years
    year = st.sidebar.selectbox('Year', filtered_papers['year'].unique())
    if year:
        st.subheader(f'Keyword Distribution Across Different Research Fields in {year}')
        plot_keyword_distribution_by_field(year)

elif analysis == 'Keyword Co-occurrence':
    st.subheader('Keyword Co-occurrence Matrix')
    plot_keyword_cooccurrence()
























