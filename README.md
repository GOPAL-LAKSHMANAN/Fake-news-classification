# Fake-news-classification

## Fake news:

#### Fake news or information disorder is false or misleading information presented as news. Fake news often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue. Although false news has always been spread throughout history, the term "fake news" was first used in the 1890s when sensational reports in newspapers were common. Nevertheless, the term does not have a fixed definition and has been applied broadly to any type of false information presented as news. It has also been used by high-profile people to apply to any news unfavorable to them. Further, disinformation involves spreading false information with harmful intent and is sometimes generated and propagated by hostile foreign actors, particularly during elections. In some definitions, fake news includes satirical articles misinterpreted as genuine, and articles that employ sensationalist or clickbait headlines that are not supported in the text. Because of this diversity of types of false news, researchers are beginning to favour information disorder as a more neutral and informative term.

## About Dataset:

#### To train a model, we need data. One popular dataset for fake news classification is available on Kaggle1. It includes two dataset which is Real and Fake news and had details such as title, text, subject, date. You can explore this dataset to build and evaluate machine learning models for detecting fake news.

#### Kaggle Dataset URL: https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection

## OUTPUT VIDEO:


https://github.com/GOPAL-LAKSHMANAN/Fake-news-classification/assets/139698710/bc49e57f-14d3-48a9-ae84-8443ff638a0a



## Machine Learning Models used:

#### 1) Logistic Regression

#### 2) Decision Tree Classification

#### 3) Gradient Boosting Classifier

#### 4) Random Forest Classifier


## Approaches:

### Importing libraries:

####    First we need to import the required libraries using IMPORT function to use there functionalities.

### Importing Dataset:

####    Select a dataset that you want to work with. Datasets are available in various formats (CSV, Excel, etc.). For this example, let’s assume you have a CSV file containing your data.Use data loading libraries (such as Pandas) to load your dataset into memory

### Explore the Data:

####    Understand the structure of your dataset by examining its columns, data types, and summary statistics.

### Preprocess the Data:

####    Handle missing values, outliers, and categorical variables. Split the dataset into training and testing subsets.

### Choose a Machine Learning Model:

####    Depending on the problem we need to select the ML model. In our project we need to use classification models to train the algorithem. Machine Learning modles we use in this project include logistic regression, decision trees, gradient boosting classifier, random forests.

### Train the Model:  
####    Fit the chosen model to the training data using the .fit() method.

### Model Evaluation:
####    Model evaluation done by classification and report was saved to .ipynb file.

### Model Saving:
####    we will save our models that we can use them for prediction purpose.

### Present Results:

####    When provided with news input, four machine learning models are capable of determining whether the given news is authentic or fabricated.
