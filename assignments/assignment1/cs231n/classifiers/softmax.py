from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****sw

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):

        scores = X[i].dot(W)
        regular = -np.max(scores)
        correct_class_score = scores[y[i]]

        cal_scores = np.exp(X[i].dot(W) + regular)
        cal_correct_class_score = np.exp(correct_class_score + regular)

        loss += -np.log(cal_correct_class_score / np.sum(cal_scores))

        mask = np.exp(scores + regular) / np.sum(np.exp(scores + regular))
        mask[y[i]] = mask[y[i]] - 1

        dW += (X[i].reshape(1, X.shape[1])).T @ (mask.reshape(1, num_classes))

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = X @ W
    regular = -np.max(scores, axis=1, keepdims=True)
    correct_class_score = scores[np.arange(0, num_train, 1), y].reshape(num_train, 1)

    cal_scores = np.exp(scores + regular)
    cal_correct_class_score = np.exp(correct_class_score + regular)

    loss += np.sum(-np.log(cal_correct_class_score / np.sum(cal_scores, axis=1, keepdims=True)))
    loss /= num_train
    loss += reg * np.sum(W * W)

    mask = np.exp(scores + regular) / np.sum(np.exp(scores + regular), axis=1, keepdims=True)
    mask[np.arange(0, num_train, 1), y] = mask[np.arange(0, num_train, 1), y] - 1

    dW += X.T @ mask
    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
