**Credits to:** Omer Sakarya[Scraper], Chris[Flask Productionization] and Ken Jee[For the project playlist]

# Data Science Salary Estimator: Project Overview 
* This Data Science project predict salaries (MAE ~ $ 10K) of data scientists.
* Scraped over 200 job descriptions from glassdoor using python and selenium.
* Cleaned the data and picked up the useful attributes.  
* Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model.
* Built a client facing API using flask.

## Code and Resources Used 
**Python Version:** 3.8
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```
**Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium    
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2
**Ken Jee:** https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

## Web Scraping
scraped over 200 job postings from `glassdoor.com`[Scraper repo is mentioned above].
The df contains the below mentioned columns:
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 

## Data Cleaning
After scraping the data, I cleaned it up so that it was usable for our model. The changes made are as follows:

*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Generated few pivot tables as well.

## Model Building 

First, I transformed the categorical variables into dummy variables.
I then slipt the data into train and tests using sklearn `train_test_split`, into 4:1.

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret .   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Linear Regression:** MAE = 18.805
*	**Lasso Regression:** MAE = 19.753
*	**Random Forest:**  MAE = 14.863

## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver by following the link mentioned above. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 
