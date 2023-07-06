import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_sETH(task, user_base, currency_relation):
    assert currency_relation != 0 and user_base != 0
    currency_relation_inverse = 1 / currency_relation**3
    sETH = np.abs(currency_relation_inverse) * (np.abs(task)**2 / np.abs(user_base)**2) 
    sETH = sigmoid(np.log(sETH))  # Use sigmoid function to keep the value within range [0,1]
    return sETH

# provided values
T = 1000  # tasks
U = 10  # users
C = 100  # ETH to USD exchange rate

# calculate scaling factor
scaling_factor = calculate_sETH(T, U, C)
print("The scaling factor is:", scaling_factor)
