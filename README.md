# Project 3: Comic character classification
My Metis project 3.

## 1. Background
In this project, I analyzed a Kaggle dataset of Marvel and DC comic book characters ([here](https://www.kaggle.com/fivethirtyeight/fivethirtyeight-comic-characters-dataset)). This dataset was originally generated for a FiveThirtyEight article about gender representation in comics ([here](https://fivethirtyeight.com/features/women-in-comic-books/)). 

My goal was to see if a character's alignment - good, bad or neutral - could be predicted using features in the dataset. This was done in two parts: cleaning and modeling. 

## 2. Cleaning and prepping features
The first step was to download and clean the data - see the Project_3_data_cleaning.ipynb notebook and Project_3_functions.py file. I downloaded data in CSV format straight from Kaggle and imported into a Jupyter notebook using Pandas. The dataset consisted of two CSV's; one for Marvel and one for DC. I combined these into a single frame and did some basic cleaning to deal with the many NaN's in the dataset. These were specific to columns; for example, NaN's in the "Hair color" column were replaced by "no color".



After some general EDA and visualization, I set my feature and target variables. **Features** were attributes of the characters, including physical features such as eye color as well as more character-description features such as Gender, whether the character had a secret identity, etc. **Target** was the character's alignment: Good, Bad or Neutral. See cleaning notebook. 

## 3. Modeling results
After my data were downloaded and cleaned, it was time to model! This project was a **supervised classification** problem, and I tried various models to classify my characters into my target classes. My best performing model was XG Boost, with the following scores. 



              precision    recall  f1-score   support

           Bad       0.61      0.50      0.55      2398
           Good      0.59      0.30      0.40      1910
           Neutral   0.40      0.06      0.11      1510
		   
		  
		  
As we can see, the recall scores are quite low, especially for the **neutral class**. 

I did notice some trends among my features and target classes. Briefly, female characters tend to be good, villains tend to have a secret identity, and red and yellow eyes tend to be associated with villainy. See my Tableau dashboard for the breakdown!

## 4. Takeaways and future work

My intuition about the low scores for the neutral class is that these characters are poorly defined (almost by definition) and they are thus difficult for a model to predict. 

Additionally, this dataset has some import drawbacks. First, the distribution of character appearances is heavily skewed: the vast majority of characters (my observations) appear only a small number of times. Further, and perhaps more importantly, this dataset lacks feature of a character's *personality* -- which are probably crucial in determining their alignment!

Thus, future work on this question should delve into more personality-level features of the characters, and strive to include only more flesh-out characters. I hope to use NLP in the next project to further investigate these characters.

## 5. Tools and techniques used
- [Jupyter](https://jupyter.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/index.html)
- [PostgreSQL](https://www.postgresql.org/)
- Classification with the following algorithms:
	- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
	- [K-nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
	- [Guassian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
	- [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
	- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
	- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
	- [XGBoost Classifier](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)

Special thanks to Brian McGarry and Richard Chiou for assistance with experimental design and implementation, and to Anterra Kennedy for help with pipeline and experimental design.
