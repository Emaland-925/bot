#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'ML\DAmmam 2020-2022.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to inspect the data
print(data.head())

# Display information about the DataFrame, including data types
print(data.info())


# In[3]:


# Check for inconsistencies in categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"Column: {column}")
    print(data[column].value_counts())
    print("\n")
# Check for missing data
missing_data = data.isnull().sum()
print(missing_data[missing_data > 0])
# Identify outliers using IQR method for numerical columns
import numpy as np

numerical_columns = data.select_dtypes(include=['int64']).columns
for column in numerical_columns:
    q1 = np.percentile(data[column], 25)
    q3 = np.percentile(data[column], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"Column: {column}, Number of outliers: {len(outliers)}")


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(nrows=len(data.select_dtypes(include=['int64', 'float64']).columns), figsize=(10, 6 * len(data.columns)))

# Iterate through each numeric column
for i, column in enumerate(data.select_dtypes(include=['int64', 'float64']).columns):
    sns.histplot(data[column], kde=True, ax=axes[i])
    axes[i].set_title(f'Univariate Analysis of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')

# Adjust layout for better spacing
plt.tight_layout()

# Show the combined plot
plt.show()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='ALLSKY_SFC_SW_DWN', y='Solar Energy', data=data)
plt.title('Scatter Plot: ALLSKY_SFC_SW_DWN vs. Solar Energy')
plt.show()

# Sort the data by Age for a clearer line
data_sorted_by_monthes = data.sort_values(by='MO')
sns.lineplot(x='MO', y='Solar Energy', data=data_sorted_by_monthes)
plt.title('Line Plot: Month vs Solar Energy')
plt.show()

# sns.boxplot(x='T2M', y='Solar Energy', data=data)
# plt.title('Box Plot: PS vs. Solar Energy')
# plt.show()

# sns.violinplot(x='ALLSKY_SFC_SW_DWN', y='Solar Energy', data=data)
# plt.title('Violin Plot: ALLSKY_SFC_SW_DWN vs. Solar Energy')
# plt.xticks(rotation=45)
# plt.show()

sns.jointplot(x='ALLSKY_SFC_SW_DWN', y='Solar Energy', data=data, kind='hex', cmap='YlGnBu')
plt.suptitle('Hexbin Plot: ALLSKY_SFC_SW_DWN vs. Solar Energy')
plt.show()

sns.pairplot(data[['CLRSKY_SFC_SW_DWN', 'Solar Energy', 'ALLSKY_SFC_SW_DWN']])
plt.suptitle('Pair Plot: CLRSKY_SFC_SW_DWN, Solar Energy, ALLSKY_SFC_SW_DWN')
plt.show()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

# Replace 'ALLSKY_SFC_SW_DWN' with 'WD10M' and 'Solar Energy' with 'Wind Energy'
sns.scatterplot(x='WD10M', y='Wind Energy', data=data)
plt.title('Scatter Plot: WD10M vs. Wind Energy')
plt.show()

data_sorted_by_monthes = data.sort_values(by='MO')
sns.lineplot(x='MO', y='Wind Energy', data=data_sorted_by_monthes)
plt.title('Line Plot: Month vs Wind Energy')
plt.show()

# sns.boxplot(x='T2M', y='Wind Energy', data=data)
# plt.title('Box Plot: PS vs. Wind Energy')
# plt.show()

# sns.violinplot(x='WD10M', y='Wind Energy', data=data)
# plt.title('Violin Plot: WD10M vs. Wind Energy')
# plt.xticks(rotation=45)
# plt.show()

sns.jointplot(x='WD10M', y='Wind Energy', data=data, kind='hex', cmap='YlGnBu')
plt.suptitle('Hexbin Plot: WD10M vs. Wind Energy')
plt.show()

sns.pairplot(data[['WS10M', 'Wind Energy', 'WD10M']])
plt.suptitle('Pair Plot: WS10M, Wind Energy, WD10M')
plt.show()


# In[7]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['ALLSKY_SFC_SW_DWN'], data['CLRSKY_SFC_SW_DWN'], data['Solar Energy'])
ax.set_xlabel('ALLSKY_SFC_SW_DWN')
ax.set_ylabel('CLRSKY_SFC_SW_DWN')
ax.set_zlabel('Solar Energy')

plt.title('3D Scatter Plot: CLRSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DWN, and Solar Energy')
plt.show()

import seaborn as sns

sns.scatterplot(x='ALLSKY_SFC_SW_DWN', y='CLRSKY_SFC_SW_DWN', size='Solar Energy', hue='MO', data=data)
plt.title('Bubble Chart: CLRSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DWN, Solar Energy, Month')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.show()

sns.pairplot(data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'Solar Energy', 'MO']], hue='MO')
plt.suptitle('Pair Plot with Color Coding: CLRSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DWN, Solar Energy, MO')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.show()


# In[8]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['WD10M'], data['WS10M'], data['Wind Energy'])
ax.set_xlabel('WD10M')
ax.set_ylabel('WS10M')
ax.set_zlabel('Wind Energy')

plt.title('3D Scatter Plot: WS10M, WD10M, and Wind Energy')
plt.show()

import seaborn as sns

sns.scatterplot(x='WD10M', y='WS10M', size='Wind Energy', hue='MO', data=data)
plt.title('Bubble Chart: WS10M, WD10M, Wind Energy, Month')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.show()

sns.pairplot(data[['WD10M', 'WS10M', 'Wind Energy', 'MO']], hue='MO')
plt.suptitle('Pair Plot with Color Coding: WS10M, WD10M, Wind Energy, MO')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
plt.show()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select only the relevant columns
selected_columns = ['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'Solar Energy', 'WD10M', 'WS10M', 'Wind Energy', 'T2M', 'RH2M', 'PS', 'UVA']
numeric_data = data[selected_columns]

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Correlation Matrix: Solar Energy, Wind Energy, T2M, RH2M, PS, UVA')
plt.show()


# In[10]:


# Scatter plot
plt.scatter(data['WD10M'], data['Wind Energy'])  # Replace 'ALLSKY_SFC_SW_DWN' with 'WD10M' and 'Solar Energy' with 'Wind Energy'
plt.title('Scatter Plot: WD10M vs. Wind Energy')
plt.xlabel('WD10M')
plt.ylabel('Wind Energy')
plt.show()

# Regression plot
sns.regplot(x='WD10M', y='Wind Energy', data=data)  # Replace 'ALLSKY_SFC_SW_DWN' with 'WD10M' and 'Solar Energy' with 'Wind Energy'
plt.title('Regression Plot: WD10M vs. Wind Energy')
plt.show()



# In[11]:


# Calculate the correlation 
corr = data.corr().abs()

top = corr[corr['Solar Energy'] > 0.5]

sorted_corr = top.sort_values(by=['Solar Energy'], ascending = False)

sorted_corr = sorted_corr['Solar Energy'].index

print('The top correlated input variables are: ',sorted_corr[1:].tolist())

print(data)


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

for feature in sorted_corr:
    # Extract the feature column
    X = data[[feature]]

    # Create a quadratic feature
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_quad = poly.fit_transform(X)

    # Fit a linear regression model
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_quad, data['Solar Energy'])

    # Extract coefficients
    best_b1 = reg.coef_[0]
    best_b2 = reg.coef_[1]
    best_b0 = reg.intercept_

    print(f'The values for B1, B2, and B0 for: [{feature}]')
    print(f'The best values for B1 is: {np.round(best_b1, 2)}')
    print(f'The best values for B2 is: {np.round(best_b2, 2)}')
    print(f'The best values for B0 is: {np.round(best_b0, 2)}')
    print('-------------------------------------------')


# In[13]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assuming data is your DataFrame
X = data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'PS', 'T2M', 'RH2M']].values
y = data.iloc[:, -1].values

# Generate Train - Test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

# The scaled data can now be used for model training and testing


# In[14]:


#OLs
from sklearn.linear_model import LinearRegression
ols_reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
ols_y = ols_reg.predict(X_test)
print('The Coefficient are: ', ols_reg.coef_)

# In[15]:


## Ridge
from sklearn.linear_model import RidgeCV

reg2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
               fit_intercept=False,cv=10).fit(X_train, y_train)

y_pred2 = reg2.predict(X_test)

print('The Coefficient are:', reg2.coef_)

# In[16]:


## Lasso

from sklearn.linear_model import LassoCV
reg3 = LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 
               fit_intercept=False,cv=10, random_state=0).fit(X_train, y_train)

y_pred3 = reg3.predict(X_test)
print('The Coefficient are:', reg3.coef_)

# In[17]:


from sklearn.metrics import mean_squared_error

ols = mean_squared_error(y_test, ols_y) 

reg_MSE = mean_squared_error(y_test, y_pred2)

lasso_MSE = mean_squared_error(y_test, y_pred3)

print('The MSE using OLS is: ', ols)
print('\n')
print('The MSE using Ridge is: ', reg_MSE)
print('\n')
print('The MSE using Lasso is: ', lasso_MSE)

# In[18]:


# Assuming linear_model is your trained Linear Regression model

# New data for prediction
new_data = np.array([[3.05, 3.74, 102.43, 18.19, 66.12]])
#X = data[['ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN', 'PS', 'T2M', 'RH2M']].values

# Make predictions on the new data using Linear Regression
predictions_linear = reg2.predict(new_data)

# Print or use the predictions as needed
print("Predictions on new data using Linear Regression:", predictions_linear)


# In[ ]:



