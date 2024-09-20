# shark_functions.py

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the Excel file
url = "https://www.sharkattackfile.net/spreadsheets/GSAF5.xls"
sharks = pd.read_excel(url, engine='xlrd')

#View all columns and rows
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# Standardize column names by removing spaces and converting to lowercase
sharks.columns = [col.replace(' ', '').lower() for col in sharks.columns]


def clean_date_column(sharks):
    """
    Clean and process the 'Date' and 'Year' columns.
    
    Args:
    sharks (pandas.DataFrame): The shark attack data.
    
    Returns:
    pandas.DataFrame: The cleaned shark attack data.
    """
    sharks['date'] = sharks['date'].astype(str).apply(lambda x: re.sub(r'[-]', ' ', x))
    sharks['date'] = pd.to_datetime(sharks['date'], errors='coerce')
    sharks['year'] = pd.to_numeric(sharks['year'], errors='coerce')
    sharks['year'] = sharks['year'].fillna(sharks['date'].dt.year)
    return sharks

def clean_type_column(sharks):
    """
    Clean and standardize the 'Type' column.
    
    Args:
    sharks (pandas.DataFrame): The shark attack data.
    
    Returns:
    pandas.DataFrame: The cleaned shark attack data.
    """
    sharks.loc[:, 'type'] = sharks['type'].str.lower().str.strip()
    sharks.loc[:, 'type'].fillna('unknown', inplace=True)
    
    type_mapping = {
        'unprovoked': 'unprovoked',
        'provoked': 'provoked',
        'boat': 'watercraft',
        'air/sea disaster': 'disaster',
        'sea disaster': 'disaster',
        'invalid': 'unknown',
        'unverified': 'unknown',
        'unconfirmed': 'unknown',
        'under investigation': 'unknown',
        'questionable': 'unknown',
        '?': 'unknown',
        'unknown': 'unknown'
    }
    
    sharks.loc[:, 'type'] = sharks['type'].replace(type_mapping)
    sharks.loc[:, 'type'] = sharks['type'].apply(lambda x: 'unknown' if x not in type_mapping.values() else x)
    
    return sharks

def assign_season(date):
    """
    Assign a season based on the month of a given date.
    
    Args:
    date (datetime): The date to assign a season to.
    
    Returns:
    str: The assigned season ('winter', 'spring', 'summer', 'fall', or 'unknown').
    """
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'fall'
    return 'unknown'

def clean_country_column(sharks):
    """
    Clean and standardize the 'Country' column.
    
    Args:
    sharks (pandas.DataFrame): The shark attack data.
    
    Returns:
    pandas.DataFrame: The cleaned shark attack data.
    """
    sharks = sharks[sharks['country'].apply(lambda x: isinstance(x, str))]
    sharks.loc[:, 'country'] = sharks['country'].str.upper()
    
    country_mapping = {
        "ST HELENA": "SAINT HELENA",
        " PHILIPPINES": "PHILIPPINES",
        "ENGLAND": "UNITED KINGDOM",
        "OKINAWA": "JAPAN",
        "AZORES": "PORTUGAL",
        "CEYLON": "SRI LANKA",
        "CRETE": "GREECE",
        " TONGA": "TONGA",
        "&": "AND",
        "ST.": "ST",
        "SRI LANKA (SRI LANKA)": "SRI LANKA",
        "ST KITTS / NEVIS": "SAINT KITTS / NEVIS",
        "ST MAARTIN": "SAINT MAARTIN"
    }
    
    for old, new in country_mapping.items():
        sharks.loc[:, 'country'] = sharks['country'].str.replace(old, new, regex=False)
    
    return sharks

def clean_shark_attack_data(sharks):
    """
    Clean the entire shark attack dataset.
    
    Args:
    sharks (pandas.DataFrame): The raw shark attack data.
    
    Returns:
    pandas.DataFrame: The cleaned shark attack data.
    """
    sharks = clean_date_column(sharks)
    sharks = clean_type_column(sharks)
    sharks = clean_country_column(sharks)
    return sharks

def preprocess(text):
    """
    Preprocess text data for clustering.
    
    Args:
    text (str): The text to preprocess.
    
    Returns:
    str: The preprocessed text.
    """
    tokens = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

def get_cluster_label(cluster, sharks, clusters):
    """
    Get a representative label for a cluster.
    
    Args:
    cluster (int): The cluster number.
    sharks (pandas.DataFrame): The shark attack data.
    clusters (numpy.ndarray): The cluster assignments.
    
    Returns:
    str: The representative activity label for the cluster.
    """
    cluster_activity = sharks.loc[clusters == cluster, 'activity']
    if cluster_activity.empty:
        return "Uncategorized"
    return cluster_activity.iloc[0]

def calculate_rates(data, sex):
    """
    Calculate attack rates for a given sex.
    
    Args:
    data (pandas.DataFrame): The shark attack data.
    sex (str): The sex to calculate rates for ('F' or 'M').
    
    Returns:
    tuple: Contains total, fatal, injury, no_injury counts and their respective rates.
    """
    total = data[data['sex'] == sex].shape[0]
    fatal = data[(data['sex'] == sex) & (data['injury'] == 'fatal')].shape[0]
    no_injury = data[(data['sex'] == sex) & (data['injury'] == 'no injury')].shape[0]
    injury = total - fatal - no_injury
    
    fatal_rate = (fatal / total) * 100
    injury_rate = (injury / total) * 100
    safe_rate = (no_injury / total) * 100
    
    return total, fatal, injury, no_injury, fatal_rate, injury_rate, safe_rate

# Main execution
if __name__ == "__main__":
    # Set pandas display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    # Load and clean data
    sharks = load_shark_data()
    sharks = clean_shark_attack_data(sharks)
    
    print("Data loaded and cleaned successfully.")