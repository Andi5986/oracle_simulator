from sklearn.linear_model import Ridge

def create_and_train_model_ridge(df):
    model_ridge = Ridge()
    model_ridge.fit(df[['Tasks', 'User Base', 'Ether to USD', 'Ether Volatility']], df['ETH'])
    return model_ridge

