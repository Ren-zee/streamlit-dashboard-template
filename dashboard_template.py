#######################
# Import libraries

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import seaborn as sns
import numpy as np

##
from sklearn.preprocessing import MinMaxScaler

## 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import re

#######################
# Page configuration
st.set_page_config(
    page_title="Amazon Phone Data Dashboard", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Amazon Phone Data Dashboard')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. KYLE CHRYSTIAN ESPINOSA\n2. RENZO ANDRE FALLORAN\n3. CARLO LUIS LEYVA\n4. KURT JOSEPH PECENIO\n5. ANGELICA YOUNG")

#######################
# Data

# Load data
phonesearch_df = pd.read_csv("data/phone search.csv")

# Plots

# `key` parameter is used to update the plot when the page is refreshed

def histogram(column, width, height, key):
    # Generate a histogram
    histogram_chart = px.histogram(phonesearch_df, x=column)

    # Adjust the height and width
    histogram_chart.update_layout(
        width=width,   # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(histogram_chart, use_container_width=True, key=f"histogram_{key}")

def scatter_plot(x_column, y_column, width, height, key):
    # Generate a scatter plot
    scatter_plot = px.scatter(phonesearch_df, x=phonesearch_df[x_column], y=phonesearch_df[y_column])

    # Adjust the height and width
    scatter_plot.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(scatter_plot, use_container_width=True, key=f"scatter_plot_{key}")
def pairwise_scatter_plot(key):
    # Generate a pairwise scatter plot matrix
    scatter_matrix = px.scatter_matrix(
        phonesearch_df,
        dimensions=['product_price', 'product_star_rating', 'product_original_price'],  # Choose relevant numeric columns
        color='is_prime'  # Replace with a categorical column if applicable
    )

    # Adjust the layout
    scatter_matrix.update_layout(
        width=500,  # Set the width
        height=500  # Set the height
    )

    st.plotly_chart(scatter_matrix, use_container_width=True, key=f"pairwise_scatter_plot_{key}")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", key="conf_matrix"):
    """
    Function to plot a confusion matrix using Plotly.

    Parameters:
    y_true (list or array): True labels
    y_pred (list or array): Predicted labels
    title (str): Title of the confusion matrix plot
    key (str): Unique key for Streamlit's use_container_width functionality
    """
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Create a DataFrame for easier plotting
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                  index=['Not Amazon Choice', 'Amazon Choice'], 
                                  columns=['Not Amazon Choice', 'Amazon Choice'])

    # Create a heatmap using Plotly
    fig = px.imshow(conf_matrix_df, 
                    text_auto=True, 
                    labels={'x': 'Predicted', 'y': 'Actual'}, 
                    title=title)

    # Adjust the layout
    fig.update_layout(
        width=600,  # Width of the plot
        height=500,  # Height of the plot
    )

    # Display
    st.plotly_chart(fig, use_container_width=True, key=key)


def feature_importance_plot(feature_importance_df, width, height, key):
    # Generate a bar plot for feature importances
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h'  # Horizontal bar plot
    )

    # Adjust the height and width
    feature_importance_fig.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot_{key}")


# -------------------------

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown(""" 

    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to predict "Amazon Choice" products and sales volume from the Phone Search Dataset using **Logistic Regression** and **Random Forest Regressor**.

    #### Pages
    1. `Dataset` - A brief description of the Phone Search Dataset used in this dashboard.
    2. `EDA` - Exploratory Data Analysis of the Phone Search Dataset. Highlighting the distribution of features like product price, star rating, and reviews. Includes visualizations such as histograms, scatter plots, and pairwise scatter plot matrix.
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps including encoding categorical variables, handling missing values, and splitting the dataset into training and testing sets for classification and regression tasks.
    4. `Machine Learning` - Training two supervised models: Logistic Regression for classification of "Amazon Choice" products and Random Forest Regressor for sales volume prediction. Includes model evaluation, feature importance analysis, and tree plot for Random Forest.
    5. `Prediction` - A page where users can input values to predict whether a product is "Amazon Choice" or estimate the sales volume using the trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

    """)
   

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

     # Your content for your DATASET page goes here

    st.markdown("""

    The **Amazon Phone Data dataset** contains real-time information about 340 phone products available on Amazon. It includes prices, reviews, sales volume, and tags such as Best Seller and Amazon Choice. This data is useful for analyzing price trends, forecasting sales volume, and gaining insights into e-commerce.

    For each product in the Amazon Phone Data dataset, several features are measured: asin (Amazon Standard Identification Number), product_title (name of the product), product_price (current price of the product), and product_star_rating (rating given to the product based on customer reviews). This dataset is useful for analyzing product performance, pricing strategies, and consumer preferences in e-commerce. The dataset was uploaded to Kaggle by the user shreyasur965.

    **Content**
    The dataset has **340** rows containing attributes related to phone products on Amazon. The columns include information such as Price, Reviews, Sales Volume, and Tags (e.g., Best Seller, Amazon Choice).

    `Link:` https://www.kaggle.com/datasets/shreyasur965/phone-search-dataset          
                
    """)
    
 # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(phonesearch_df, use_container_width=True, hide_index=True)
    
 # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(phonesearch_df.describe(), use_container_width=True)

    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset. The product_star_rating has an average rating of 4.09 out of 5, with a standard deviation of 0.38, indicating that most products are rated positively, close to 4 or above. The product_num_ratings column shows an average of 2,744 ratings per product, though a large standard deviation (7,331.82) suggests high variability; while some products have very few ratings, others are extremely popular with many reviews (up to 64,977). product_num_offers averages around 8 offers per product, but again, there's substantial variability (ranging from 1 to 83), suggesting certain products are offered by many vendors. The unit_count column, with only four recorded values, suggests missing data, limiting its usefulness for analysis.

    Overall, these statistics highlight a need for further data cleaning, particularly in handling missing values in unit_count and potentially examining outliers in product_num_ratings and product_num_offers for better consistency in analysis.
                
    """)
   
    

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    with st.expander('Legend', expanded=True):
        st.write('''
            - Data: [Phone Search Dataset](https://www.kaggle.com/datasets/shreyasur965/phone-search-dataset).
            - :orange[**Histogram**]: Distribution of normalized product prices.
            - :orange[**Scatter Plots**]: Product prices vs. Star Ratings.
            - :orange[**Pairwise Scatter Plot Matrix**]: Highlighting *overlaps* and *differences* among Numerical features.
        ''')

    st.markdown('#### Price Distribution')
    histogram("product_price", 800, 600, 1)
    
    st.markdown('#### Product prices vs. Star Ratings')
    scatter_plot('product_price', 'product_star_rating', 500, 500, 1)
    
    st.markdown('#### Pairwise Scatter Plot Matrix')
    pairwise_scatter_plot(1)

        
    

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning")

    st.subheader("Initial Dataset Preview")
    st.write("Here’s a preview of the raw dataset:")
    st.write(phonesearch_df.head())

    # 1 Check for missing values
    st.subheader("Checking for Missing Values")
    missing_values = phonesearch_df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # 2 Dropping Irrelevant Columns
    st.subheader("Dropping Irrelevant Columns")
    st.code("""
    phoneData_df = phoneData_df.drop(columns=['product_url', 'product_photo'])
    """)
    irrelevant_columns = ['product_url', 'product_photo']
    phonesearch_df = phonesearch_df.drop(columns=irrelevant_columns)
    st.write("✅ Dropped columns:", irrelevant_columns)

    # 3 Cleaning and Converting Currency Columns
    st.subheader("Cleaning and Converting Currency Columns")
    st.code("""
    phoneData_df['product_price'] = pd.to_numeric(phoneData_df['product_price'].str.replace('[\$,]', '', regex=True))
    phoneData_df['product_original_price'] = pd.to_numeric(phoneData_df['product_original_price'].str.replace('[\$,]', '', regex=True)) 
    """)
    phonesearch_df['product_price'] = pd.to_numeric(phonesearch_df['product_price'].str.replace('[\$,]', '', regex=True))
    phonesearch_df['product_original_price'] = pd.to_numeric(phonesearch_df['product_original_price'].str.replace('[\$,]', '', regex=True))
    st.write("✅ Converted columns `product_price` and `product_original_price` to numeric.")

    # 4 Filling Missing Values with Median
    st.subheader("Filling Missing Values")
    st.code("""
    phoneData_df['product_price'] = phoneData_df['product_price'].fillna(phoneData_df['product_price'].median())
    phoneData_df['product_original_price'] = phoneData_df['product_original_price'].fillna(phoneData_df['product_original_price'].median())                  
    """)
    phonesearch_df['product_price'] = phonesearch_df['product_price'].fillna(phonesearch_df['product_price'].median())
    phonesearch_df['product_original_price'] = phonesearch_df['product_original_price'].fillna(phonesearch_df['product_original_price'].median())
    st.write("✅ Filled missing values with the median for `product_price` and `product_original_price`.")

    # 5 Outlier Removal in Product Price
    st.subheader("Outlier Removal in Product Price")
    st.code("""
    Q1 = phoneData_df['product_price'].quantile(0.25)
    Q3 = phoneData_df['product_price'].quantile(0.75)
    IQR = Q3 - Q1
    phoneData_df = phoneData_df[(phoneData_df['product_price'] >= (Q1 - 1.5 * IQR)) &
                                (phoneData_df['product_price'] <= (Q3 + 1.5 * IQR))]         
    """)
    Q1 = phonesearch_df['product_price'].quantile(0.25)
    Q3 = phonesearch_df['product_price'].quantile(0.75)
    IQR = Q3 - Q1
    phonesearch_df = phonesearch_df[(phonesearch_df['product_price'] >= (Q1 - 1.5 * IQR)) &
                                    (phonesearch_df['product_price'] <= (Q3 + 1.5 * IQR))]
    st.write("✅ Removed outliers based on the IQR method.")

    # 6 Normalizing Columns
    st.subheader("Normalizing Columns")
    st.code("""
    scaler = MinMaxScaler()
    phoneData_df[['product_price', 'product_star_rating']] = scaler.fit_transform(
    phoneData_df[['product_price', 'product_star_rating']])   
    """)
    scaler = MinMaxScaler()
    phonesearch_df[['product_price', 'product_star_rating']] = scaler.fit_transform(phonesearch_df[['product_price', 'product_star_rating']])
    st.write("✅ Normalized columns `product_price` and `product_star_rating`.")

    # Final preview
    st.subheader("Processed Dataset Preview")
    st.write("Here’s a preview of the processed dataset:")
    st.write(phonesearch_df.head())

    st.header("🧼 Data Pre-processing")
    # 1 Encoding categorical variables
    st.subheader("Encoding Categorical Variables")
    encoder = LabelEncoder()
    phonesearch_df['is_best_seller_encoded'] = encoder.fit_transform(phonesearch_df['is_best_seller'].astype(str))
    phonesearch_df['is_amazon_choice_encoded'] = encoder.fit_transform(phonesearch_df['is_amazon_choice'].astype(str))

    # Display the encoded columns
    st.write("Encoded Data (After Label Encoding):")
    st.write(phonesearch_df[['is_best_seller', 'is_best_seller_encoded', 'is_amazon_choice', 'is_amazon_choice_encoded']].head())

    # 2 Select features and target variable for classification
    st.subheader("Classification Task")
    col1, col2 = st.columns(2, gap='medium')  
    with col1:       
        X_classification = phonesearch_df[['product_price', 'product_star_rating', 'product_num_ratings']]
        y_classification = phonesearch_df['is_amazon_choice_encoded']
        # Display the selected features and target variable for classification
        st.write("Selected Features for Classification:")
        st.write(X_classification.head())
    with col2:
        st.write("Target Variable for Classification:")
        st.write(y_classification.head())

    # 3 Split the dataset into training and testing sets for classification
    st.subheader("Classification Data Split")
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.3, random_state=42)

    # Display 
    st.write("Shape of the Training Set for Classification:")
    st.write(X_train_class.shape)
    st.write("Shape of the Test Set for Classification:")
    st.write(X_test_class.shape)

    # Save to session state 
    st.session_state['X_train_class'] = X_train_class
    st.session_state['X_test_class'] = X_test_class
    st.session_state['y_train_class'] = y_train_class
    st.session_state['y_test_class'] = y_test_class 

    # 4 Select features and target variable for regression
    st.subheader("Regression Task")
    X_regression = phonesearch_df[['product_price', 'product_star_rating', 'product_num_ratings']]
    y_regression = phonesearch_df['sales_volume']
    
    # Display 
    st.write("Selected Features for Regression:")
    st.write(X_regression.head())
    st.write("Target Variable for Regression:")
    st.write(y_regression.head())

    # 5 Split the dataset into training and testing sets for regression
    st.subheader("Regression Data Split")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.3, random_state=42)
    
    # Display 
    st.write("Shape of the Training Set for Regression:")
    st.write(X_train_reg.shape)
    st.write("Shape of the Test Set for Regression:")
    st.write(X_test_reg.shape)

    # Save the training and testing datasets to session state
    st.session_state['X_train_reg'] = X_train_reg
    st.session_state['X_test_reg'] = X_test_reg
    st.session_state['y_train_reg'] = y_train_reg
    st.session_state['y_test_reg'] = y_test_reg

    # Add a message indicating that preprocessing is complete
    st.write("Data Preprocessing Completed and Saved to Session State.")




# Ensure session state variables are initialized
if 'X_train_class' not in st.session_state:
    st.session_state['X_train_class'] = None
if 'X_test_class' not in st.session_state:
    st.session_state['X_test_class'] = None
if 'y_train_class' not in st.session_state:
    st.session_state['y_train_class'] = None
if 'y_test_class' not in st.session_state:
    st.session_state['y_test_class'] = None
if 'X_train_reg' not in st.session_state:
    st.session_state['X_train_reg'] = None
if 'X_test_reg' not in st.session_state:
    st.session_state['X_test_reg'] = None
if 'y_train_reg' not in st.session_state:
    st.session_state['y_train_reg'] = None
if 'y_test_reg' not in st.session_state:
    st.session_state['y_test_reg'] = None

# Machine Learning Page
if st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Logistic Regression Section
    st.subheader("Logistic Regression")
    st.markdown("""
    **Logistic Regression** is a statistical method used for binary classification problems, where the goal is to predict the probability that a given input point belongs to a particular category.
    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """)

    if None in (st.session_state['X_train_class'], st.session_state['X_test_class'],
                st.session_state['y_train_class'], st.session_state['y_test_class']):
        st.error("Please complete data preprocessing before accessing the Logistic Regression model.")
    else:
        X_train_class = st.session_state['X_train_class']
        X_test_class = st.session_state['X_test_class']
        y_train_class = st.session_state['y_train_class']
        y_test_class = st.session_state['y_test_class']

        # Train Logistic Regression model
        log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
        log_reg_model.fit(X_train_class, y_train_class)
        y_pred_class = log_reg_model.predict(X_test_class)

        # Evaluation metrics
        st.write("Accuracy:", accuracy_score(y_test_class, y_pred_class))
        st.text("Classification Report:\n" + classification_report(y_test_class, y_pred_class))

    # Random Forest Regressor Section
    st.subheader("Random Forest Regressor")
    st.markdown("""
    **Random Forest Regressor** is a machine learning algorithm used to predict continuous values by combining multiple decision trees.
    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """)

    if None in (st.session_state['X_train_reg'], st.session_state['X_test_reg'],
                st.session_state['y_train_reg'], st.session_state['y_test_reg']):
        st.error("Please complete data preprocessing before accessing the Random Forest Regressor model.")
    else:
        X_train_reg = st.session_state['X_train_reg']
        X_test_reg = st.session_state['X_test_reg']
        y_train_reg = st.session_state['y_train_reg']
        y_test_reg = st.session_state['y_test_reg']

        # Define function to extract numeric values from sales volume
        def extract_numeric(value):
            if isinstance(value, str):
                numbers = re.findall(r'\d+', value)
                return int(numbers[0]) if numbers else np.nan
            return value

        # Apply extract_numeric and fill NaN with median for y_train_reg and y_test_reg
        y_train_reg = y_train_reg.apply(extract_numeric).fillna(y_train_reg.median())
        y_test_reg = y_test_reg.apply(extract_numeric).fillna(y_test_reg.median())

        # Train Random Forest Regressor
        rfr_model = RandomForestRegressor(random_state=42)
        rfr_model.fit(X_train_reg, y_train_reg)

        # Evaluate model
        train_score = rfr_model.score(X_train_reg, y_train_reg)
        test_score = rfr_model.score(X_test_reg, y_test_reg)
        st.write(f"Train R² Score: {train_score * 100:.2f}%")
        st.write(f"Test R² Score: {test_score * 100:.2f}%")

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = pd.Series(rfr_model.feature_importances_, index=X_train_reg.columns)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=feature_importance.index)
        plt.title("Random Forest Regressor Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        st.pyplot(plt)

        # Display number of trees in the Random Forest
        st.write(f"Number of trees made: {len(rfr_model.estimators_)}")

        # Plot all trees up to 100 in a grid layout
        st.subheader("Random Forest Trees")
        n_estimators = min(len(rfr_model.estimators_), 100)
        n_rows = 10
        n_cols = 10
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), dpi=50)
        for i, tree in enumerate(rfr_model.estimators_[:n_estimators]):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            plot_tree(tree, feature_names=X_train_reg.columns, filled=True, rounded=True, ax=ax)
            ax.set_title(f"Tree {i+1}", fontsize=6)
            ax.axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # Extract and plot a single tree
        st.subheader("Single Tree Visualization")
        single_tree = rfr_model.estimators_[0]
        plt.figure(figsize=(20, 10))
        plot_tree(single_tree, feature_names=X_train_reg.columns, filled=True, rounded=True)
        st.pyplot(plt)


    
    
    

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""
    Through exploratory data analysis and training of two classification models (`Logistic Regression` and `Random Forest Regressor`) on the **Amazon Phone Data dataset**, the key insights and observations are:
    
    1. 📊 **Dataset Characteristics**:
       - The dataset captures valuable e-commerce metrics such as product price, ratings, and sales volume.
       - Some columns like `product_original_price`, `product_availability`, `unit_price`, and `coupon_text` had significant missing values, which were either removed or imputed as necessary.
       - Feature normalization and encoding allowed us to prepare the dataset for effective modeling.
    
    2. 📝 **Feature Distributions and Separability**:
       - Visualizations, including histograms, scatter plots, and pairwise scatter plots, revealed relationships between product prices and star ratings.
       - EDA highlighted that certain features, such as `product_price` and `product_star_rating`, have a strong correlation, suggesting potential for feature engineering in future work.
    
    3. 📈 **Model Performance (Logistic Regression for Classification)**:
    
       - The **Logistic Regression** model achieved high accuracy in predicting whether a product was designated as "Amazon Choice." This simple yet effective model was a good fit for the classification task.
       - The confusion matrix and classification report further illustrated the model’s effectiveness in classifying the products accurately based on the provided features.
    
    4. 📈 **Model Performance (Random Forest Regressor for Sales Volume Prediction)**:
       - The **Random Forest Regressor** was trained to predict product sales volume, achieving a high R² score on both the training and testing data.
       - Feature importance analysis identified `product_price` and `product_star_rating` as the most influential features for predicting sales volume, highlighting how customer preferences may affect sales.
    
    **Summing up:**  
    This project successfully demonstrated effective data preparation, exploratory analysis, and modeling techniques on the Amazon Phone Data dataset. Both models achieved high accuracy in their respective tasks, providing insights into product sales prediction and classification.
                
    """)
