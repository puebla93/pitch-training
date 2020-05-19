import numpy as np
import scipy
from util.utils import Obj, QuadraticLeastSquaresModel, Ball
import matplotlib.pyplot as plt

params = Obj(
    debugging=False,
    # a model that can be fitted to data points
    model=QuadraticLeastSquaresModel(),
    # the minimum number of data values required to fit the model
    n=3,
    # the maximum number of iterations allowed in the algorithm
    max_iters=100,
    # a threshold value for determining when a data point fits a model
    eps=1e3,
    # a percent of close data values required to assert that a model fits well to data
    closeData_percent=9./10
)

def ransac(data):
    """fit model parameters to data using the RANSAC algorithm
    Params:
        data - a set of observed data points
    Return:
        bestdata - best fit data by the model(or nil if no good model is found)"""

    # num_closeData - the number of close data values required to assert that a model fits well to data
    num_closeData = data.shape[0] * params.closeData_percent - params.n
    # num_closeData = data.shape[0] - params.n
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None

    while iterations < params.max_iters:
        maybe_idxs, test_idxs = random_partition(data.shape[0])
        maybeinliers = data[maybe_idxs]
        test_points = data[test_idxs]

        maybemodel = params.model.fit(maybeinliers)
        test_err = params.model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < params.eps] # select indices of rows with accepted points
        alsoinliers = data[also_idxs]

        if params.debugging:
            plot_debug(data, maybe_idxs, also_idxs, maybemodel)
        
        if len(alsoinliers) >= num_closeData:
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
    else:
        bestdata = data[best_inlier_idxs]
        bestdata = list(bestdata)
        bestdata.sort(key=lambda p: p[0])
        bestdata = fit_points(bestdata, bestfit)
        return bestdata, bestfit

def random_partition(n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.random.permutation(n_data)
    idxs1 = all_idxs[:params.n]
    idxs2 = all_idxs[params.n:]
    return idxs1, idxs2

def fit_points(points, fit):
    func = params.model.func
    Yfit, Zfit = fit
    new_points = [[point[0], func(point[0], Yfit[0], Yfit[1], Yfit[2]), func(point[0], Zfit[0], Zfit[1], Zfit[2]), point[3]] for point in points]
    return np.array(new_points)

def plot_debug(data, maybe_idxs, also_idxs, maybemodel):
    # print('test_err.min()', test_err.min())
    # print('test_err.max()', test_err.max())
    # print('np.mean(test_err)', np.mean(test_err))
    # print('iteration %d : len(alsoinliers) = %d'%(iterations, len(alsoinliers)))
    # print()

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    maybeYmodel, maybeZmodel = maybemodel

    maybeinliers_points = data[maybe_idxs]
    maybeinliers_x, maybeinliers_y = maybeinliers_points[:, 0], maybeinliers_points[:, 1]

    alsoinliers_points = data[also_idxs]
    alsoinliers_x, alsoinliers_y = alsoinliers_points[:, 0], alsoinliers_points[:, 1]

    func_x = np.arange(0, 1024, dtype='int')
    # func_x = np.arange(x.min()-10, x.max()+10, dtype='int')
    func_y = [params.model.func(x, maybeYmodel[0], maybeYmodel[1], maybeYmodel[2]) for x in func_x]
    func_y = np.array(func_y)
    func_z = [params.model.func(x, maybeZmodel[0], maybeZmodel[1], maybeZmodel[2]) for x in func_x]
    func_z = np.array(func_z)

    plt.scatter(x, y, c='w', edgecolors='r', label='data')
    plt.scatter(x, z, c='w', edgecolors='r', label='data')
    plt.scatter(alsoinliers_x, alsoinliers_y, edgecolors='b', label='alsoinliers')
    plt.scatter(maybeinliers_x, maybeinliers_y, c='g', edgecolors='g', label='maybeinliers')
    plt.plot(func_x, func_y, 'm', lw=2, label='maybeYmodel')
    plt.plot(func_x, func_z, 'm', lw=2, label='maybeZmodel')

    plt.xlim(0, 1024)
    plt.ylim(0, 600)
    # plt.xlim(x.min()-10, x.max()+10)
    # plt.ylim(y.min()-10, y.max()+10)
    plt.legend()
    plt.show()

def setUp(nparams):
    params.setattr(nparams)

if __name__ == '__main__':
    pass
