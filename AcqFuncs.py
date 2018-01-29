import scipy.stats as stats


class ExpImpr:

    def __init__(self, xi=0.01, bounds=None, fmin=None):
        """
        Constructor for ExpImpr class (Expected Improvement acquisition)

        Args :
            xi (float) : Exploration encouragement parameter
            bounds (tuple) : Bounds on the dompain of the objective functions
            fmin (float) : Minimum attained so far
        """
        self.fmin = fmin
        self.xi = xi
        self.name = "EI"
        #Should acquisition be maximized of minimized
        self.opti_way = "max"
        self.bounds = bounds

    def set_fmin(self, fmin):
        """
        Change fmin stored parameter

        Args :
            fmin (float)
        """
        self.fmin = fmin

    def set_bounds(self, bounds):
        self.bounds = bounds

    def evaluate(self, y, sigma):
        if sigma == 0:
            return 0.0
        else:
            z = (self.fmin - y - self.xi) / sigma
            value = (self.fmin - y - self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            return max(0, value)


class LowConfBound:

    def __init__(self, eta=2):
        self.eta = eta
        #In minimization case, LCB should be minimized not maximized
        self.opti_way = "max"
        self.name = "LCB"

    def evaluate(self, y, sigma):
        return - (y - self.eta * sigma)