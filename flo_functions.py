import numpy as np
# may turn this into a class....



meta = {
    "sigmoid": {
        "params": ["mu", "sigma", "y0", "y1"],
        "params_tex": ["\\mu", "\\sigma", "y_{{0}}", "y_{{1}}"],
        "params_unit": ["µs", "µs", "", ""],
        "formula": "$\\frac{{(y_1 - y_0)}}{{1+\\frac{{exp((\\mu-x)}}{{sigma}}}} + y_0$",
    }
}






def sigmoid(x, mu=0, sigma=1, y0=0, y1=1):
    '''
    a simple sigmoid function based on 1/(1+exp(x))
    
    centre of slpoe: mu
    width of slope: sigma (~central 46%)
    f(mu-sigma) = 0.269
    y0: start value of funciton
    y1: end value of function
    
    '''
    return(
        (y1-y0)/(1+(np.exp((mu-np.array(x))/sigma)))+y0
    )


def gaus(x, A=1, mu=0, sigma=1):
    return(
        A * np.exp(-(np.array(x)-mu)**2 / (2*sigma)**2)
    )