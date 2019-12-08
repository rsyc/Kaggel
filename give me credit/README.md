#---Give me some credits---

The code in this repository is in Python (Main.py).

## Project description

Having some features of customers, we want to see if credit card(s) should be given to a person or not.  

## The data

Data was provided as csv files (cs_tes.csv, cs_training.csv). Each file contains 12 columns:
'>Unnamed'  --> customer ID

TRAIN/TEST DEPENDENT VALUE: 
>'SeriousDlqin2yrs' --> Delinquent customer (Y/N)

FEATURES:
> 'RevolvingUtilizationOfUnsecuredLines' --> debt to credit limit
> 'age' --> customer age
> 'NumberOfTime30-59DaysPastDueNotWorse' --> #of time 30-59 days have past due date
> 'DebtRatio' --> debt to income ratio
> 'MonthlyIncome' --> monthly income 
> 'NumberOfOpenCreditLinesAndLoans' --> #of open credits
> 'NumberOfTimes90DaysLate' --> #of time more than 90 days have past due date
> 'NumberRealEstateLoansOrLines' --> #of mortgages
> 'NumberOfTime60-89DaysPastDueNotWorse' --> #of time 60-89 days have past due date
> 'NumberOfDependents' --> #of dependent persons (children and spouse)

## Solution

### Data Analysis

(1) Analysis on nulls to see if nan values in the features are indictive for serious delinquency rate.
(2) Grouped data based on delinquency or not-delinquency and plotted results for each feature to see the difference.
(3) Seemed that delays (either short=30-59 or long >90 days) have a great impact on defining a delinquent person so a new parameter was defined that adds these three together.
(4) Age distribution shows the role it plays in delinquency (30-60 years seem to be more likely for delinquency)

### Modeling Technique

used Logestic regression model (sklearn,linear_model,LogisticRegression)
As the test csv file did not have y_test, I split the training data into training and test to see how the model works and then used th model on test csv data for prediction
Model performance was checked using consusion matrix, computing accuracy, precision, recall, and f1 scores. 

### Results

Model output (prediction for test features) is saved into a csv file name: 'results.csv'





