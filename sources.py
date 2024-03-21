import numpy as np
import datetime as dt
from zoneinfo import ZoneInfo
import so3g.proj as proj
import ephem
from matplotlib import pyplot as plt

so3gsite = proj.coords.SITES['so']
site = so3gsite.ephem_observer()
CHILE = ZoneInfo("America/Santiago")
sources = {
    'moon':  ephem.Moon,
    'jupiter': ephem.Jupiter,
    'venus': ephem.Venus,
    'saturn': ephem.Saturn,
    'mars': ephem.Mars,
    'uranus': ephem.Uranus,
    # 'neptune': obj = ephem.Neptune()
    # 'mercury': obj = ephem.Mercury()
}

def sources_angles(start, end, site=site, zone=CHILE):

    if not isinstance(start, dt.datetime):
        start_list = start.split('/')
        start = dt.datetime(int(start_list[0]), int(start_list[1]), int(start_list[2]), int(start_list[3]), int(start_list[4]), tzinfo=zone)

    if not isinstance(end, dt.datetime):
        end_list = end.split('/')
        end = dt.datetime(int(end_list[0]), int(end_list[1]), int(end_list[2]), int(end_list[3]), int(end_list[4]), tzinfo=zone)

    delta = dt.timedelta(days=1)

    # Calculate position every day
    sun_distance = {'t': [], }
    for k in sources.keys():
        sun_distance[k] = []
    current_time = start

    # sun distance
    while current_time <= end:
        site.date = ephem.Date(current_time)

        sun_distance['t'].append(current_time)
        for k, source in sources.items():
            sun = ephem.Sun(site)
            obj = source(site)
            angle = meas_angle(sun.az, sun.alt, obj.az, obj.alt)
            sun_distance[k].append(angle)

        current_time += delta

    # Calculate maximum el every day
    max_el = {'t': [], }
    for k in sources.keys():
        max_el[k] = []

    current_time = start
    while current_time <= end:
        max_el['t'].append(current_time)
        for k, source in sources.items():
            els = []
            for i in range(100):
                site.date = ephem.Date(current_time + dt.timedelta(days=i/100.))
                els.append(np.rad2deg(source(site).alt))
            max_el[k].append(np.max(els))
        current_time += dt.timedelta(days=1)
    return sun_distance, max_el

def plot_sources_angles(start, end, thre=49, site=site, zone=CHILE):
    """
    Args:
        delta (day)
        thre (deg)
    """
    sun_distance, max_el = sources_angles(start, end, site=site, zone=zone)

    fig, ax = plt.subplots(2, 1, figsize=(9, 9), sharex=True, gridspec_kw=dict(bottom=.2, right=.85, hspace=0) )
    for k in sources:
        ax[0].plot(sun_distance['t'], sun_distance[k], lw=1., alpha=.7, label=k)
        ax[1].plot(max_el['t'], max_el[k], lw=1., alpha=.7, label=k)
    ax[0].axhline(y=thre, color='r', linestyle='-')

    ax[0].set_ylabel('delta Angle between Sun [deg]')
    ax[1].set_ylabel('maximul elevation [deg]')
    ax[1].set_xlabel('Time (Local)')
    plt.xticks(rotation=45, ha='right')
    ax[0].legend(bbox_to_anchor=(1.2, 1), loc='upper right')
    plt.show()

def meas_angle(az1, el1, az2, el2):

    x1 = np.cos(el1)*np.sin(az1)
    y1 = np.cos(el1)*np.cos(az1)
    z1 = np.sin(el1)

    x2 = np.cos(el2)*np.sin(az2)
    y2 = np.cos(el2)*np.cos(az2)
    z2 = np.sin(el2)

    dot = x1*x2 + y1*y2 + z1*z2
    return np.rad2deg(np.arccos(dot))
