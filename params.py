class Obj(object):
    def __init__(self, **kwarg):
        for k, v in kwarg.items():
            self.__setattr__(k, v)
    def __repr__(self):
        return str(self.__dict__)
    def __str__(self):
        return str(self.__dict__)

params = Obj(
    thresh_blockSize=31,
    max_percentArea=10,
    min_percentArea=1,
    percentSideRatio=20,
    numberOfSizes=5,
    useHull=True,
    useKmeans=False,
    kmeans_k=6,
    medianBlur_ksize=5,
    max_percentRadius=10,
    min_percentRadius=4
)
