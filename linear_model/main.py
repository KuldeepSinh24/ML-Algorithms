from LinearRegress import LinearRegress
from LMS import LMS
import numpy as np
import matplotlib.pyplot as plt

def main():    
    X = np.linspace(0,10, 11).reshape((-1, 1))
    T = -2 * X + 3.2
    print("X : {}".format(str(X)))
    print("T : {}".format(str(T)))
    print("Training the LinearRegress model...")
    ls = LinearRegress()

    ls.train(X, T)
    print("LS Predictions : {}".format(str(ls.use(X))))

    plt.plot(ls.use(X))

    print("Traiing the LMS model...")
    lms = LMS(0.1)
    lms.train(X, T)
    plt.plot(lms.use(X))
    print("LMS Predictions : {}".format(str(lms.use(X))))
    
    print("==========END===============")

main()