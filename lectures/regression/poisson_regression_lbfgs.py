import jax.numpy as jnp
import numpy as np
import modin.pandas as pd
from distributed import Client
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from jax import random, jit
import math

if __name__ == '__main__':
    client = Client()
    df = pd.read_csv("/workspaces/data_mining/data_mining/aiml-common/lectures/regression/data/nyc_bb_bicyclist_counts.csv", header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])

    # add columns that have been extracted from the date column
    ds = df.index.to_series()
    df['MONTH'] = ds.dt.month
    df['DAY_OF_WEEK'] = ds.dt.dayofweek
    df['DAY'] = ds.dt.day

    # select the wanted columns
    df = df[['DAY_OF_WEEK', 'PRECIP', 'HIGH_T', 'LOW_T', 'BB_COUNT']]
    
    print(df.head())
    
    # TODO: add another categorical column ('DAY_OF_WEEK_CATEGORY') that will mark the day of the week as weekend, friday and workday (3 categories)
    # Read the excellent article: https://towardsdatascience.com/beware-of-the-dummy-variable-trap-in-pandas-727e8e6b8bde
    # df = pd.get_dummies(df, columns=['DAY_OF_WEEK_CATEGORY'], drop='if_binary')
    # print(df.head())
    
    # select the columns for x
    x_columns = ['DAY_OF_WEEK', 'PRECIP', 'HIGH_T', 'LOW_T']
    
    
    # split the data into train and test
    df_train, df_test = train_test_split(df, test_size=0.2)
    x_train = df_train[x_columns].values.astype(jnp.float32)
    y_train = df_train['BB_COUNT'].values.astype(jnp.float32)
    x_test = df_test[x_columns].values.astype(jnp.float32)
    y_test = df_test['BB_COUNT'].values.astype(jnp.float32)
    
    # number of examples
    m = df.shape[0]
    m_train = df_train.shape[0]
    m_test = df_test.shape[0]
    
    scaler = StandardScaler()
    x_train_scaled  = scaler.fit_transform(x_train)
    x_train_mu = scaler.mean_ 
    x_train_sigma = scaler.scale_
    x_train_scaled = pd.DataFrame(x_train_scaled)
    
    
    x_test_scaled  = scaler.fit_transform(x_test)
    x_test_mu = scaler.mean_ 
    x_test_sigma = scaler.scale_
    x_test_scaled = pd.DataFrame(x_test_scaled)
    
    # solution with LBFGS optimizer
    model = PoissonRegressor(alpha=0.0, max_iter=1000, tol=1e-6, verbose=1)
    model.fit(x_train_scaled, y_train)
    y_hat_test_lbfgs = model.predict(x_test_scaled)
    score = model.score(x_test_scaled, y_test)
    print('Score: {}'.format(score))
    
    plt.figure()
    plt.title('Poisson regression with LBFGS Solver: Test dataset predictions')
    plt.plot(y_hat_test_lbfgs)
    plt.plot(y_test)
    plt.legend(['y_hat', 'y'])       
    plt.savefig('/workspaces/data_mining/data_mining/aiml-common/lectures/regression/images/poisson-regression-test-predictions-lbfgs.png')
    
