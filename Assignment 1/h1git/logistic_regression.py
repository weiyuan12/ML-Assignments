import numpy as np
from h1_util import numerical_grad_check

def logistic(z):
    """ 
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    
    np.exp may come in handy
    Args:
        z: numpy array shape (d,) 
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function 
    """
    logi = np.zeros(z.shape)
    ### YOUR CODE HERE
    logi = 1/(1+np.exp(-z))
    ### END CODE
    assert logi.shape == z.shape
    return logi


class LogisticRegressionClassifier():

    def __init__(self):
        self.w = None

    def cost_grad(self, X, y, w):
        """
        Compute the average negative log likelihood and gradient under the logistic regression model 
        using data X, targets y, weight vector w 
        
        np.log, np.sum, np.choose, np.dot may be useful here
        Args:
           X: np.array shape (n,d) float - Features 
           y: np.array shape (n,)  int - Labels 
           w: np.array shape (d,)  float - Initial parameter vector

        Returns:
           cost: scalar: the average negative log likelihood for logistic regression with data X, y 
           grad: np.array shape(d, ) gradient of the average negative log likelihood at w 
        """
        cost = 0
        grad = np.zeros(w.shape)
        ### YOUR CODE HERE
        grad = - 1/len(y) * np.matmul(logistic(- np.dot(X, w)* y) *y, X)
        cost = 1/len(y) * np.sum(np.log(1 + np.exp(- np.dot(X, w) * y)))
        
        ### END CODE
        assert grad.shape == w.shape
        return cost, grad


    def fit(self, X, y, w=None, lr=0.1, batch_size=16, epochs=10):
        if w is None: w = np.zeros(X.shape[1])
        history = []        
        ### YOUR CODE HERE 
        for i in range(epochs):
            perm = np.random.permutation(len(y))
            X = X[perm]
            y = y[perm]
            for j in range(0, len(y), batch_size):
                X_b = X[j:j+batch_size]
                y_b = y[j:j+batch_size]
                #Getting gradient
                cost, grad = self.cost_grad(X_b, y_b, w)
                w = w - lr*grad
            #Getting cost
            cost, grad = self.cost_grad(X, y, w)    
            history.append(cost)           
        ### END CODE
        self.w = w
        self.history = history


    def predict(self, X):
        """ Classify each data element in X.

        Args:
            X: np.array shape (n,d) dtype float - Features 
        
        Returns: 
           p: numpy array shape (n, ) dtype int32, class predictions on X (-1, 1). NOTE: We want a class here, 
           not a probability between 0 and 1. You should thus return the most likely class!

        """
        out = np.ones(X.shape[0])
        ### YOUR CODE HERE
        out = logistic(np.dot(X, self.w.T))
        for i in range (len(out)):
            if out[i]>=0.5:
                out[i] = 1
            else:
                out[i] = -1
        ### END CODE
        return out
    
    def score(self, X, y):
        """ Compute model accuracy  on Data X with labels y

        Args:
            X: np.array shape (n,d) dtype float - Features 
            y: np.array shape (n,) dtype int - Labels 

        Returns: 
           s: float, number of correct predictions divided by n. NOTE: This is accuracy, not in-sample error!

        """
        s = 0
        ### YOUR CODE HERE
        correct_pred  = np.sum(self.predict(X) == y)
        s = correct_pred/len(y)
        ### END CODE
        return s
        

    
def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

    
def test_cost():
    print('*'*5, 'Testing Cost Function')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([-1, -1, 1], dtype='int64')
    w = np.array([0.0, 0.0])
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost,_ = lr.cost_grad(X, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

    
def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])    
    w = np.array([0.0, 0.0])
    y = np.array([-1, -1, 1]).astype('int64')
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: lr.cost_grad(X, y, w=z)
    numerical_grad_check(f, w)
    print('Test Success')


    
if __name__ == '__main__':
    test_logistic()
    test_cost()
    test_grad()
    
    
