class Obj(object):
    def __init__(self, **kwarg):
        for k, v in kwarg:
            self.__setattr__(k, v)
    def __repr__(self):
        return str(self.__dict__)
    def __str__(self):
        return str(self.__dict__)

params = Obj(
    
)
