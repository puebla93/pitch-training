import numpy as np
import scipy
from utils import Obj, QuadraticLeastSquaresModel

params = Obj(
    debugging=False,
    # a model that can be fitted to data points
    model=QuadraticLeastSquaresModel(),
    # the minimum number of data values required to fit the model
    n=3,
    # the maximum number of iterations allowed in the algorithm
    max_iters=1000,
    # a threshold value for determining when a data point fits a model
    eps=7e3
)

def ransac(data, d, return_all=False):
    """fit model parameters to data using the RANSAC algorithm
    Params:
        data - a set of observed data points
        d - the number of close data values required to assert that a model fits well to data
    Return:
        bestfit - model parameters which best fit the data (or nil if no good model is found)"""

    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None

    while iterations < params.max_iters:
        maybe_idxs, test_idxs = random_partition(params.n, data.shape[0])
        # maybe_idxs, test_idxs = random_partition(params.n, data)
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]

        maybemodel = params.model.fit(maybeinliers)
        test_err = params.model.get_error(test_points, maybemodel)

        also_idxs = test_idxs[test_err < params.eps] # select indices of rows with accepted points
        alsoinliers = data[also_idxs, :]

        if params.debugging:
            print 'test_err.min()', test_err.min()
            print 'test_err.max()', test_err.max()
            print 'np.mean(test_err)', np.mean(test_err)
            print 'iteration %d : len(alsoinliers) = %d'%(iterations, len(alsoinliers))

        if len(alsoinliers) >= d - params.n:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = params.model.fit(betterdata)
            better_errs = params.model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)

            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

        iterations+=1

    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        # return bestfit, {'inliers':best_inlier_idxs}
        return bestfit, best_inlier_idxs
    else:
        return bestfit

def random_partition(n, n_data):
# def random_partition(n, data):
    """return n random rows of data (and also the other len(data)-n rows)"""

    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]

    return idxs1, idxs2

    all_idxs = np.arange(data.shape[0])
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    maybeinliers = data[idxs1]
    test_points = data[idxs2]

    inliers = []
    for points in maybeinliers:
        all_idxs = np.arange(points.shape[0])
        np.random.shuffle(all_idxs)
        idxs1 = all_idxs[0]
        idxs2 = all_idxs[1:]

    return inliers, test_points

if __name__ == '__main__':
    pass
