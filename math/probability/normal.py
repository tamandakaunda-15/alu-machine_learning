#!/usr/bin/env python3
"""
Normal Distribution Class

This module contains the Normal class which represents a normal distribution
and provides methods to calculate the z-score, x-value, and PDF of
the distribution.
"""


class Normal:
    """
    A class that represents a normal distribution.

    Attributes:
        mean (float): The mean of the distribution.
        stddev (float): The standard deviation of the distribution.

    Methods:
        __init__(self, data=None, mean=0., stddev=1.):
            Initializes the Normal distribution, either from given data or
            using specified mean and standard deviation values.

        z_score(self, x):
            Calculates the z-score for a given x-value.

        x_value(self, z):
            Calculates the x-value for a given z-score.

        pdf(self, x):
            Calculates the value of the PDF for a given x-value.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution.

        If data is provided, calculates the mean and standard deviation from
        the data. Otherwise, uses the given mean and standard deviation.

        Args:
            data (list, optional): A list of data points to estimate the
 distribution.
            mean (float, optional): The mean of the distribution
 (default is 0).
            stddev (float, optional): The standard deviation of
 the distribution (default is 1).

        Raises:
            ValueError: If stddev is less than or equal to 0.
            TypeError: If data is not a list.
            ValueError: If data does not contain multiple values.
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean manually
            self.mean = sum(data) / len(data)

            # Calculate standard deviation manually
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """
        Calculates the z-score for a given x-value.

        Args:
            x (float): The value to calculate the z-score for.

        Returns:
            float: The z-score for the given x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value for a given z-score.

        Args:
            z (float): The z-score to calculate the x-value for.

        Returns:
            float: The x-value corresponding to the given z-score.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value.

        Args:
            x (float): The x-value to calculate the PDF for.

        Returns:
            float: The PDF value for the given x.
        """
        # Calculate the exponent term
        exponent = -0.5 * ((x - self.mean) ** 2) / (self.stddev ** 2)
        # Calculate the normalization factor
        normalization = 1 / (self.stddev * (2 * 3.141592653589793) ** 0.5)
        # Return the PDF value
        return normalization * (2.718281828459045 ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value.

        Args:
            x (float): The x-value to calculate the CDF for.

        Returns:
            float: The CDF value for the given x.
        """
        # Calculate the z-score for the given x
        z = self.z_score(x)

        # Use the approximation for the error function (erf)
        t = 1.0 / (1.0 + 0.3275911 * abs(z))
        erf_approx = 1 - (0.254829592 * t - 0.284496736 * t**2 +
                          1.421413741 * t**3 - 1.453152027 * t**4 +
                          1.061405429 * t**5) * (2.718281828459045 ** (-z * z / 2))
        
        # CDF = 0.5 * (1 + erf(z / sqrt(2)))
        return 0.5 * (1 + erf_approx)
