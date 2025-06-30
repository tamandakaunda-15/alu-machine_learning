#!/usr/bin/env python3
"""
Uses the GitHub API to print the location of a specific user,
where user is passed as first argument of the script with full API URL.

Example:
    ./2-user_location.py https://api.github.com/users/holbertonschool
"""

import requests
from sys import argv
from time import time


if __name__ == "__main__":
    if len(argv) < 2:
        raise TypeError(
            "Input must have the full API URL passed as an argument:\n"
            'Example: "./2-user_location.py '
            'https://api.github.com/users/holbertonschool"'
        )

    try:
        url = argv[1]
        response = requests.get(url)

        if response.status_code == 403:
            reset = response.headers.get('X-Ratelimit-Reset')
            wait_time = int(reset) - time()
            minutes = round(wait_time / 60)
            print('Reset in {} min'.format(minutes))
        else:
            data = response.json()
            location = data.get('location')
            if location:
                print(location)
            else:
                print('Not found')

    except Exception:
        print('Not found')
