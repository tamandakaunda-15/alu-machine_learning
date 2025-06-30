#!/usr/bin/env python3
"""
This module defines the `availableShips` function which
queries the Star Wars API (SWAPI) and returns a list of
starships that can carry at least a given number of passengers.
"""

import requests


def availableShips(passengerCount):
    """
    Returns a list of starships from the Star Wars API
    that can carry at least `passengerCount` passengers.

    Args:
        passengerCount (int): The minimum number of passengers
                              the ship must be able to carry.

    Returns:
        List of ship names (list of strings).
        Empty list if no ships are found.
    """
    url = 'https://swapi.dev/api/starships/'
    matching_ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break

        data = response.json()
        for ship in data.get('results', []):
            passengers = ship.get('passengers', '0').replace(',', '').split()[0]
            try:
                if passengers.isdigit() and int(passengers) >= passengerCount:
                    matching_ships.append(ship.get('name'))
            except ValueError:
                continue

        url = data.get('next')

    return matching_ships
