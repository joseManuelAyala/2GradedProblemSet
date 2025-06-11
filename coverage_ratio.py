import numpy as np

def coverage_ratio(returns, var_array):
    """
        Fraction of times r_t < VaR_t.
        Expect coverage ~ alpha if VaR_t is the alpha-quantile (negative).
    """

    returns = np.asarray(returns)
    var_array = np.asarray(var_array)

    violations = returns < var_array
    coverage = np.mean(violations)

    return coverage
