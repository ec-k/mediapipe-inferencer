import time
from functools import partial, wraps

class ThrottleDecorator:
    def __init__(self, func, interval):
        self.func = func
        self.interval = interval
        self.last_time = 0

    def __get__(self,obj,objtype=None):
        if obj is None:
            return self.func
        return partial(self,obj)

    def __call__(self, *args, **kwargs):
        now = time.time()
        if now - self.last_time > self.interval:
            self.last_time = now
            return self.func(*args, **kwargs)

def throttle(interval):
    def applyDecorator(func):
        decorator = ThrottleDecorator(func=func, interval=interval)
        return wraps(func)(decorator)
    return applyDecorator