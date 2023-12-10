import pandas as pd
import os
import torch
from sklearn.preprocessing import StandardScaler

# umap of the dataset

def get_house_dataset(config):        
    #load dataset
    file = os.path.join(os.path.dirname(__file__), 'house.csv')
    house_data = pd.read_csv(file)

    #view percentage of nan values per column and sort it by descending order
    null_cols = house_data.isnull().sum().sort_values(ascending=False)/len(house_data)*100
    #drop all columns with more than 50% of nan values
    house_data = house_data.drop(null_cols[null_cols>50].index, axis=1)

    sale_price = house_data['SalePrice']
    #find the median of the sale_price variable
    median = sale_price.median()
    #create a target variable with 1 if the price is above the median and 0 if it is below
    target = [1 if i > median else 0 for i in sale_price]
    #add the target variable to the dataset
    house_data['>163k'] = target
    #create a new column with the the sale price divided in quintiles and label each quantile by the range of the sale price
    house_data['quintile_range'] = pd.qcut(house_data['SalePrice'], 5, labels=['0-129k', '129k-163k', '163k-214k', '214k-280k', '280k-755k'])
    #display the distribution of the sale price by quintile
    house_data['quintile_range'].value_counts().plot(kind='bar')

    features = house_data.columns.drop(['SalePrice', '>163k', 'quintile_range', 'Id'])

    #retrieve all columns that are categorical
    categorical_cols = [col for col in features if house_data[col].dtype == 'object']
    #transform all categorical columns into numerical columns
    for col in categorical_cols:
        house_data[col] = pd.Categorical(house_data[col]).codes

    #delete rows with nan values
    house_data = house_data.dropna()

    #convert the dataset into a tensor
    scaler = StandardScaler()
    X = house_data[features]
    X = scaler.fit_transform(X)
    y = house_data['>163k']
    y_s = house_data['quintile_range'].astype('category').cat.codes

#     X = torch.tensor(X.values).float()
#     y = torch.tensor(y.values).float()
#     y_s

#     #scale the dataset
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

    return X, y.to_numpy(), y_s.to_numpy()

# print(house_data())