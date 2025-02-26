#!/usr/bin/env python3


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
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution.

        If data is provided, calculates the mean and standard deviation from
        the data. Otherwise, uses the given mean and standard deviation.

        Args:
            data (list, optional): A list of data points to estimate the
                                    distribution.
            mean (float, optional): The mean of the distribution (default is 0).
            stddev (float, optional): The standard deviation of the distribution
                                      (default is 1).

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
