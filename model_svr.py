from sklearn.svm import SVR

def create_and_train_model(df):
    model = SVR()
    model.fit(df[['Tasks', 'User Base', 'Ether to USD']], df['ETH'])

    return model
