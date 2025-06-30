#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the upcoming launch as:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

The “upcoming launch” is the one which is the soonest from now, in UTC.
If 2 launches have the same date, it's the first one in the API result.
"""

import requests


if __name__ == "__main__":
    url = "https://api.spacexdata.com/v4/launches/upcoming"

    try:
        response = requests.get(url)
        launches = response.json()
    except Exception:
        print("Error fetching launch data")
        exit(1)

    date_check = float('inf')
    launch_name = None
    rocket_id = None
    launchpad_id = None
    launch_date = None

    for launch in launches:
        launch_date_unix = launch.get('date_unix')
        if launch_date_unix and launch_date_unix < date_check:
            date_check = launch_date_unix
            launch_date = launch.get('date_local')
            launch_name = launch.get('name')
            rocket_id = launch.get('rocket')
            launchpad_id = launch.get('launchpad')

    rocket_name = "Unknown Rocket"
    launchpad_name = "Unknown Pad"
    launchpad_location = "Unknown Location"

    if rocket_id:
        try:
            rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(
                rocket_id)
            rocket_data = requests.get(rocket_url).json()
            rocket_name = rocket_data.get('name', rocket_name)
        except Exception:
            pass

    if launchpad_id:
        try:
            pad_url = "https://api.spacexdata.com/v4/launchpads/{}".format(
                launchpad_id)
            pad_data = requests.get(pad_url).json()
            launchpad_name = pad_data.get('name', launchpad_name)
            launchpad_location = pad_data.get('locality', launchpad_location)
        except Exception:
            pass

    print("{} ({}) {} - {} ({})".format(
        launch_name, launch_date, rocket_name,
        launchpad_name, launchpad_location
    ))
