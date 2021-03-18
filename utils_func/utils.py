
"""
功能函数
"""
import pandas as pd
import time
import inspect


# flat = lambda lst: sum(map(flat, lst), []) if hasattr(lst, '__iter__') else [lst]  # python2
def flat(lst):
    if not hasattr(lst, '__iter__') or type(lst) in [str]:
        return [lst]
    else:
        return sum(map(flat, lst), [])


def F3_groupby(df, by):
    return pd.DataFrame.groupby(df, by, as_index=False, sort=False, group_keys=False)

def time_it(logger=None):
    def outer(func):
        def inner(*args, **kwds):

            tic = time.time()
            res = func(*args, **kwds)
            toc = time.time()

            fn = inspect.getfile(func).split('/')[-1]
            msg = ('%s --------> time on %s: %.2f mins' %
                   (fn, func.__name__, (toc - tic) / 60))

            if logger is not None:
                logger.debug(msg)
            else:
                print(msg)
            return res

        return inner

    return outer


def timer(func):
    def wrapper(*args, **kwds):
        tic = time.time()
        res = func(*args, **kwds)
        toc = time.time()

        fn = inspect.getfile(func).split('/')[-1]
        msg = ('%s --------> time on %s: %.2f seconds' %
               (fn, func.__name__, toc - tic))
        print(msg)
        return res

    return wrapper