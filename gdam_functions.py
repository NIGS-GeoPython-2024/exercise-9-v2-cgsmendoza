import numpy as np

def linregress(x, y):

    #convert x, y to numpy arrays
    x = np.array(x) 
    y = np.array(y)

    # assign variables
    delta = (len(x) * (x**2).sum()) - (x.sum()**2)
    A = (((x**2).sum() * y.sum()) - (x.sum() * (y*x).sum())) / delta
    B = ((len(x)*(y*x).sum()) - (x.sum() * y.sum())) / delta
    return A, B

def pearson(x, y):
    x = np.array(x)
    y = np.array(y)
    
    # assign variables for the mean of x, y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x-x_mean)**2) * np.sum((y-y_mean)**2))

    # calculate Pearson correlation coefficient
    r = numerator / denominator
    
    return r

def chi_squared(obs, exp, std):
    N = len(obs)
    chi = (np.sum(((obs - exp) ** 2) / (std ** 2))) / N
    return chi