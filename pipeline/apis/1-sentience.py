#!/usr/bin/env python3
"""
Defines a method to query the Star Wars API and return the list of
home planets for all sentient species.
"""

import requests


def sentientPlanets():
    """
    Uses the Star Wars API to return the list of home planets
    for all sentient species.

    Returns:
        list: Names of home planets of sentient species.
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    species_list = []
    while url:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                break
            results = response.json()
            species_list += results.get('results', [])
            url = results.get('next')
        except requests.RequestException:
            break

    home_planets = []
    for species in species_list:
        if (species.get('designation') == 'sentient' or
                species.get('classification') == 'sentient'):
            homeworld_url = species.get('homeworld')
            if homeworld_url:
                try:
                    planet = requests.get(homeworld_url).json()
                    planet_name = planet.get('name')
                    if planet_name:
                        home_planets.append(planet_name)
                except requests.RequestException:
                    continue

    return home_planets
