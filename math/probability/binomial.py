import numpy as np

class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        # If data is provided, calculate n and p from data
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Calculate p and n from the data
            p = np.mean(data) / n  # This gives the average success rate
            n = round(np.mean(data))  # Round n to the nearest integer
        else:
            # If no data is given, use n and p
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

        # Set instance attributes
        self.n = n
        self.p = p
