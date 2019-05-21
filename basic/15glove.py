from functools import wraps


# https://github.com/hans/glove.py/blob/master/util.py
# http://www.foldl.me/2014/glove-python/

def listify(fn):
    """
    Use this decorator on a generator function to make it return a list
    instead.
    """

    @wraps(fn)
    def listified(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return listified