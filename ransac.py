import numpy as np
import scipy
from utils import Obj, QuadraticLeastSquaresModel, Ball

params = Obj(
    debugging=False,
    # a model that can be fitted to data points
    model=QuadraticLeastSquaresModel(),
    # the minimum number of data values required to fit the model
    n=4,
    # the maximum number of iterations allowed in the algorithm
    max_iters=1000,
    # a threshold value for determining when a data point fits a model
    eps=1e4
)

def ransac(data):
    """fit model parameters to data using the RANSAC algorithm
    Params:
        data - a set of observed data points
    Return:
        bestdata - best fit data by the model(or nil if no good model is found)"""

    # d - the number of close data values required to assert that a model fits well to data
    d = data.shape[0] - params.n
    iterations = 0
    bestfit = None
    besterr = np.inf
    bestdata = None

    while iterations < params.max_iters:
        maybeinliers, test_points = random_partition(data)

        maybemodel = params.model.fit(maybeinliers)
        test_err = params.model.get_error(test_points, maybemodel)

        alsoinliers = test_points[test_err < params.eps]

        if params.debugging:
            print 'test_err.min()', test_err.min()
            print 'test_err.max()', test_err.max()
            print 'np.mean(test_err)', np.mean(test_err)
            print 'iteration %d : len(alsoinliers) = %d'%(iterations, len(alsoinliers))
        
        if len(alsoinliers) >= d:
            betterdata = np.concatenate((maybeinliers, alsoinliers))
            bettermodel = params.model.fit(betterdata)
            better_errs = params.model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)

            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                bestdata = betterdata

        iterations+=1

    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    else:
        bestdata = list(bestdata)
        bestdata.sort(key=lambda p: p.center[0])
        bestdata = fitted_balls(bestdata, bestfit)
        return bestdata, bestfit

def random_partition(data):
    """return params.n random points of data (and also the other len(data)-n poimts)"""
    all_idxs = np.arange(data.shape[0])
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:params.n]
    idxs2 = all_idxs[params.n:]

    test_points = []
    for points in data[idxs2]:
        test_points = np.concatenate((test_points, points))

    maybeinliers = []
    for points in data[idxs1]:
        all_idxs = np.arange(points.shape[0])
        np.random.shuffle(all_idxs)
        idxs1 = all_idxs[:1]
        idxs2 = all_idxs[1:]
        maybeinliers = np.concatenate((maybeinliers, points[idxs1]))
        test_points = np.concatenate((test_points, points[idxs2]))

    maybeinliers = list(maybeinliers)
    maybeinliers.sort(key=lambda p: p.center[0])
    return np.array(maybeinliers), test_points

def fitted_balls(balls, fit):
    func = params.model.func
    new_balls = map(lambda b: Ball(np.array([b.center[0],
                                    func(b.center[0], fit[0], fit[1], fit[2])]),
                                    b.radius), balls)
    return np.array(new_balls)

if __name__ == '__main__':
    pass
