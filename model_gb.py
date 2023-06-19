from sklearn.ensemble import GradientBoostingRegressor

def create_and_train_model_gb(df):
    model_gb = GradientBoostingRegressor()
    model_gb.fit(df[['Tasks', 'User Base', 'Ether to USD', 'Ether Volatility']], df['ETH'])
    return model_gb
