# -*- coding: utf-8 -*-
import json
import logging
import logging.handlers
import traceback
from time import strftime
import os
from datetime import datetime, date, timedelta
# workaround non-ascii characters
import sys
from functools import wraps

f = sys.stdout
reload(sys)
sys.stdout = f # workaround ipython switching to kerner console
sys.setdefaultencoding('utf8')


# init logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# L = logging.getLogger()
# L.addHandler(logging.handlers.RotatingFileHandler("logs/" + strftime("%Y-%m-%d") + '.log',
#                                                   mode='a', maxBytes=2 ** 20, backupCount=1000))
# L.handlers[1].setFormatter(L.handlers[0].formatter)


def toTuple(x):
        if type(x) is tuple:
            return x
        else:
            return (x,)


def getIfDefined(d, *keys):
    ptr = d
    for k in keys:
        if type(ptr) is list:
            if k < len(ptr):
                ptr = ptr[k]
                continue
            else:
                return None
        if hasattr(ptr, '__iter__') and k in ptr:
            ptr = ptr[k]
        else:
            return None
    return ptr

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def jdumps(o):
    import json
    return json.dumps(o, ensure_ascii=0, indent=2, sort_keys=1)


def filterAlnum(s):
    """ Suppresses all non-alphanumercals in a string """
    r = [c for c in s]
    for i, c in enumerate(r):
        if not c.isalnum():
            r[i] = '_'
    return "".join(r)


def to_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def logged_function(f):
    """
    logged_function(f) -> None\n
    Decorator, logs call and exit of a function.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            logging.debug(traceback.format_exc())
            return None

    return wrapper
