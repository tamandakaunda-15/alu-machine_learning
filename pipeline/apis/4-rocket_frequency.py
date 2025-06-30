#!/usr/bin/env python3
"""
Uses the (unofficial) SpaceX API to print the number of launches per rocket as:
<rocket name>: <number of launches>

The list is ordered by the number of launches in descending order.
If rockets have the same number, they are ordered alphabetically.
"""

import requests


if __name__ == "__main__":
    try:
        launches = requests.get("https://api.spacexdata.com/v4/launches").json()
        rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()
    except Exception:
        print("Error fetching data")
        exit(1)

    # Build a rocket ID-to-name mapping
    rocket_id_to_name = {r['id']: r['name'] for r in rockets}

    # Count launches per rocket
    rocket_counts = {}
    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket_name = rocket_id_to_name.get(rocket_id)
        if rocket_name:
            rocket_counts[rocket_name] = rocket_counts.get(rocket_name, 0) + 1

    # Sort by number of launches (desc), then name (asc)
    sorted_rockets = sorted(
        rocket_counts.items(),
        key=lambda x: (-x[1], x[0])
    )

    for name, count in sorted_rockets:
        print(f"{name}: {count}")
