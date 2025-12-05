# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

df = pd.read_csv('../data/diabetes.csv')
df.head()


# %%
# cleaning data - check for missing values
df.isnull().sum()

# %%
print('The data set has ' + str(len(df)) + ' rows')
print('The data set has ' + str(len(df.columns)) + ' columns')

# %%
# no missing values!!
# Summary of each variable of interest:

# Target variable: Diabetes_012
df['Diabetes_012'].value_counts()

# %%
# Smoker
df['Smoker'].value_counts()

# %%
# PhysActivity
df['PhysActivity'].value_counts()

# %%
# Fruits
df['Fruits'].value_counts()

# %%
# Veggies
df['Veggies'].value_counts()

# %%
# HvyAlcoholConsump
df['HvyAlcoholConsump'].value_counts()

# %%
# Education
df['Education'].value_counts()
# df['Education'].describe()

# %%
# Income
df['Income'].value_counts()
# df['Income'].describe()

# %%
# AnyHealthcare
df['AnyHealthcare'].value_counts()

# %%
# Sex
df['Sex'].value_counts()

# %%
# Age
df['Age'].value_counts()
# df['Age'].describe()

# %%
# BMI
df['BMI'].describe()

# %%
# moving onto RQ1: What lifestyle factors (physical activity, smoking, drinking, etc.) have the biggest influence in developing diabetes?
# ok figure this out tmrw???
factors = ['Smoker', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump']

# 1. calculate percentage of each group (nos and yes) that has been diagnosed with diabetes
percent_no = []
percent_yes = []

for factor in factors:
    no_group = df[df[factor] == 0]
    yes_group = df[df[factor] == 1]

    percentage_no = (no_group['Diabetes_012'] == 2).mean() * 100
    percentage_yes = (yes_group['Diabetes_012'] == 2).mean() * 100

    percent_no.append(percentage_no)
    percent_yes.append(percentage_yes)

# Step 2: Plot
x = np.arange(len(factors))
width = 0.3

plt.figure(figsize=(10, 6))

# gives us two bars per feature (learned this from geeks for geeks)
plt.bar(x - width/2, percent_no, width, label='0 (No)')
plt.bar(x + width/2, percent_yes, width, label='1 (Yes)')

plt.title('Diabetes Risk based on Lifestyle Factor')
plt.xlabel('Lifestyle Factor')
plt.ylabel('Percentage Diagnosed with Diabetes per Lifestyle Factor')
plt.xticks(x, factors)
plt.legend(title='Response to Lifestyle Factor')
plt.text(0.5, -0.15, "This barplot indicates the prevalence of diabetes for individuals who answered yes or no to each lifestyle factor.", ha='center', va='center', transform=plt.gca().transAxes)
plt.show()

# %%
# challenge tasks 1: ML
df['Diabetes_Binary'] = df['Diabetes_012'].apply(lambda x: 1 if x == 2 else 0)
X = df[factors]
y = df['Diabetes_Binary']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# use random state and stratify to make sure we 1. get the same results everytime, 2. make sure that not all of one type
# is funneled into the training set and vice versa --> i.e maintains consistency (can prob get rid of random state?
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y)

# Create and fit the model - need to balance it because too much of our data set is healthy individuals (therefor not predicting
# diabetic individuals)
logreg_model = LogisticRegression(class_weight='balanced')
logreg_model.fit(X_train, y_train)

# Predictions and basic evaluation - coefficients + odds ratio
y_pred = logreg_model.predict(X_test)

# use this to show how our model classfies
# print(classification_report(y_test, y_pred, zero_division=0))

# checking coefficients of our logreg model
# coefficient > 0 increase odds of being diabetic
# odds ratio > 1: high risk
# odds ratio < 1: lower risk (factor is a protective one?)
# = 1: no effect
coefficients = pd.DataFrame({
    'Factor': X.columns,
    # gets actual coef of factors
    'Coefficient': logreg_model.coef_[0],
    # finds odds ratio based on coef by doing np.exp on the coef of each feature (converts it into an
    # interpretable percentage (either over 1 or less than 1, meaning the feature is more likely to affect or decrease effect)
    'Odds Ratio': np.exp(logreg_model.coef_[0])
})

print(coefficients.sort_values(by='Odds Ratio', ascending=False))

# %%
# random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(class_weight='balanced')
rf_model.fit(X_train, y_train)

rf_ypreds = rf_model.predict(X_test)
# print(classification_report(y_test, rf_ypreds, zero_division=0))

factor_importance = rf_model.feature_importances_
importances = pd.DataFrame({
    'Factor': X.columns,
    'Importance': factor_importance
})

print(importances.sort_values(by='Importance', ascending=False))

# %%
# result validity: chi square
import scipy.stats as stats

for factor in factors:
    contingency_table = pd.crosstab(df[factor], df['Diabetes_Binary'])
    chi_squared, p_val, _, expected = stats.chi2_contingency(contingency_table)
    print(factor)
    print('Chi-squared value: ' + str(chi_squared))
    print('p-value: ' + str(p_val))
    print('expected values: ' + str(expected))
    print()

# null: these variables are independent 
# alternate: these variables are dependent
# p < 0.05: statistical significant evidence

# %%
#  rq2: Do socioeconomic factors correlate to an increased risk of developing diabetes?

# Income
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Income', y='Diabetes_Binary', errorbar=None)
plt.title('Diabetes Prevalence Based on Income Levels')
plt.xlabel('Income Levels')
plt.ylabel('Proportion of Diabetic Individuals')
plt.xticks(ticks=range(8),
           labels=['<$10k', '$10–15k', '$15–20k', '$20–25k',
                   '$25–35k', '$35–50k', '$50–75k', '>$75k'],
           rotation=30
)
plt.text(0.5, -0.25, "This plot describes the distribution of people who are diabetic in each income group.", ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# %%
# Education
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Education', y='Diabetes_Binary', errorbar=None)
plt.title('Diabetes Prevalence Based on Education Levels')
plt.xlabel('Education Levels')
plt.ylabel('Proportion of Diabetic Individuals')
plt.xticks(ticks=range(6),
           labels=['Never attended', 'Elementary', 'Some highschool',
                   'Highschool graduate', 'Some college',
                   'College graduate'],
           rotation=30
)
plt.text(0.5, -0.35, "This plot describes the distribution of people who are diabetic in each education level group.", ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# %%
# AnyHealthcare
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='AnyHealthcare', y='Diabetes_Binary', errorbar=None)
plt.title('Diabetes Prevalence Based on Having Healthcare or Not')
plt.xlabel('Healthcare Status')
plt.ylabel('Proportion of Diabetic Individuals')
plt.xticks(ticks=range(2),
           labels=['No healthcare', 'Has healthcare'],
           rotation=30
)
plt.text(0.5, -0.35, "This plot describes the distribution of people who are diabetic depending on if they have healthcare or not.", ha='center', va='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()

# %%
# challenge task: ML
X_rq2 = df[['Income', 'Education', 'AnyHealthcare']]
y_rq2 = df['Diabetes_Binary']

# use random state and stratify to make sure we 1. get the same results everytime, 2. make sure that not all of one type
# is funneled into the training set and vice versa --> i.e maintains consistency
X_rq2_train, X_rq2_test, y_rq2_train, y_rq2_test = train_test_split(
    X_rq2, y_rq2, test_size=0.3, stratify=y_rq2)

# Create and fit the model - need to balance it because too much of our data set is healthy individuals (therefor not predicting
# diabetic individuals)
logreg_model_rq2 = LogisticRegression(class_weight='balanced')
logreg_model_rq2.fit(X_rq2_train, y_rq2_train)

# Predictions and basic evaluation - coefficients + odds ratio
y_rq2_pred = logreg_model_rq2.predict(X_rq2_test)

# use this to show how our model classfies
# print(classification_report(y_rq2_test, y_rq2_pred, zero_division=0))

# checking coefficients of our logreg model
# coefficient > 0 increase odds of being diabetic
# odds ratio > 1: high risk
# odds ratio < 1: lower risk (factor is a protective one?)
# = 1: no effect
coefficients_rq2 = pd.DataFrame({
    'Factor': X_rq2.columns,
    # gets actual coef of factors
    'Coefficient': logreg_model_rq2.coef_[0],
    # finds odds ratio based on coef by doing np.exp on the coef of each feature (converts it into an
    # interpretable percentage (either over 1 or less than 1, meaning the feature is more likely to affect or decrease effect)
    'Odds Ratio': np.exp(logreg_model_rq2.coef_[0])
})

print(coefficients_rq2.sort_values(by='Odds Ratio', ascending=False))

# %%
# random forest:
rf_model_rq2 = RandomForestClassifier(class_weight='balanced')
rf_model_rq2.fit(X_rq2_train, y_rq2_train)

rf_ypreds_rq2 = rf_model_rq2.predict(X_rq2_test)
# print(classification_report(y_rq2_test, rf_ypreds_rq2, zero_division=0))

factor_importance_rq2 = rf_model_rq2.feature_importances_
importances_rq2 = pd.DataFrame({
    'Factor': X_rq2.columns,
    'Importance': factor_importance_rq2
})

print(importances_rq2.sort_values(by='Importance', ascending=False))

# %%
# challenge task: result validity

# chi-squared test on anyhealthcare bc it's a categorical variable
contingency_table_rq2 = pd.crosstab(df['AnyHealthcare'], df['Diabetes_Binary'])
chi_squared_rq2, p_val_rq2, _, expected = stats.chi2_contingency(contingency_table_rq2)
print()
print('Chi-squared value: ' + str(chi_squared_rq2))
print('p-value: ' + str(p_val_rq2))
print(expected)

#  null is always independent
# use pval of 0.05

# %%
# result validity continued:
# on income and education -> correlation test
# use spearmanr because good for qualitative ordinal values with quantitative variable

from scipy.stats import spearmanr

socio_factors_ord = ['Income', 'Education']
for factor in socio_factors_ord:
    correlation_rq2, _ = spearmanr(df[factor], df['Diabetes_Binary'])
    print(factor)                  
    print('Correlation: ' + str(correlation_rq2))
    print()

# %%
# rq3: Are there any demographic factors that are associated with the likelihood of being diagnosed with diabetes?

# Age
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Age', y='Diabetes_Binary', errorbar=None)

plt.title('Diabetes Prevalence by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Proportion with Diabetes')

plt.xticks(ticks=range(13),
           labels=['18–24', '25–29', '30–34', '35–39', '40–44', '45–49',
                   '50–54', '55–59', '60–64', '65–69', '70–74', '75–79', '80+'],
           rotation=30)

plt.text(0.5, -0.25, "This plot shows the proportion of people who are diabetic in each age group.", ha='center', va='center', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()

# %%
# Sex
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Sex', y='Diabetes_Binary', errorbar=None)
plt.title('Diabetes Prevalence by Sex')
plt.xlabel('Sex')
plt.ylabel('Proportion with Diabetes')
plt.xticks(ticks=range(2),
           labels=['Female', 'Male'],
           rotation=30
)
plt.text(0.5, -0.25, "This plot shows the proportion of people who are diabetic in each sex group.", ha='center', va='center', transform=plt.gca().transAxes)
plt.show()

# %%
# BMI
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Diabetes_012', y='BMI')
plt.title('BMI Distribution by Diabetes Diagnosis')
plt.xlabel('Diabetes Diagnosis')
plt.ylabel('Body Mass Index (BMI)')
plt.xticks(ticks=range(3),
           labels=['Healthy', 'Pre-Diabetic', 'Diabetic'],
           rotation=30
)
plt.text(0.5, -0.25, "This plot shows the BMI of people in each diabetes diagnosis group.", ha='center', va='center', transform=plt.gca().transAxes)
plt.show()

# %%
# challenge task: ML
X_rq3 = df[['Age', 'Sex', 'BMI']]
y_rq3 = df['Diabetes_Binary']

# use random state and stratify to make sure we 1. get the same results everytime, 2. make sure that not all of one type
# is funneled into the training set and vice versa --> i.e maintains consistency
X_rq3_train, X_rq3_test, y_rq3_train, y_rq3_test = train_test_split(
    X_rq3, y_rq3, test_size=0.3, stratify=y)

# Create and fit the model - need to balance it because too much of our data set is healthy individuals (therefor not predicting
# diabetic individuals)
logreg_model_rq3 = LogisticRegression(class_weight='balanced')
logreg_model_rq3.fit(X_rq3_train, y_rq3_train)

# Predictions and basic evaluation - coefficients + odds ratio
y_rq3_pred = logreg_model_rq3.predict(X_rq3_test)

# use this to show how our model classfies
# print(classification_report(y_rq3_test, y_rq3_pred, zero_division=0))

# checking coefficients of our logreg model
# coefficient > 0 increase odds of being diabetic
# odds ratio > 1: high risk
# odds ratio < 1: lower risk (factor is a protective one?)
# = 1: no effect
coefficients_rq3 = pd.DataFrame({
    'Factor': X_rq3.columns,
    # gets actual coef of factors
    'Coefficient': logreg_model_rq3.coef_[0],
    # finds odds ratio based on coef by doing np.exp on the coef of each feature (converts it into an
    # interpretable percentage (either over 1 or less than 1, meaning the feature is more likely to affect or decrease effect)
    'Odds Ratio': np.exp(logreg_model_rq3.coef_[0])
})

print(coefficients_rq3.sort_values(by='Odds Ratio', ascending=False))


# %%
# random forest

rf_model_rq3 = RandomForestClassifier(class_weight='balanced')
rf_model_rq3.fit(X_rq3_train, y_rq3_train)

rf_ypreds_rq3 = rf_model_rq3.predict(X_rq3_test)
# print(classification_report(y_rq3_test, rf_ypreds_rq3, zero_division=0))

factor_importance_rq3 = rf_model_rq3.feature_importances_
importances_rq3 = pd.DataFrame({
    'Factor': X_rq3.columns,
    'Importance': factor_importance_rq3
})

print(importances_rq3.sort_values(by='Importance', ascending=False))

# %%
# result validity

correlation_rq3, _ = spearmanr(df['Age'], df['Diabetes_Binary'])
print('Age')                  
print('Correlation: ' + str(correlation_rq3))
print()

# %%
# t-test for BMI

import scipy.stats as stats

bmi_no_diabetes = df[df['Diabetes_Binary'] == 0]['BMI']
bmi_diabetes = df[df['Diabetes_Binary'] == 1]['BMI']

t_stat, p_val_bmi = stats.ttest_ind(bmi_no_diabetes, bmi_diabetes, equal_var=False)
print('T-test statistic: ' + str(t_stat))
print('p-value: ' + str(p_val_bmi))

# %%
# t-test for sex
sex_no_diabetes = df[df['Diabetes_Binary'] == 0]['Sex']
sex_diabetes = df[df['Diabetes_Binary'] == 1]['Sex']

t_stat_sex, p_val_sex = stats.ttest_ind(sex_no_diabetes, sex_diabetes, equal_var=False)
print('T-test statistic: ' + str(t_stat_sex))
print('p-value: ' + str(p_val_sex))

# %%



