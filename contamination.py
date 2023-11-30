import datetime as dt
import numpy as np
from zoneinfo import ZoneInfo
import so3g.proj as proj
import ephem

so3gsite = proj.coords.SITES['so']
site = so3gsite.ephem_observer()
CHILE = ZoneInfo("America/Santiago")

beam_thre = np.deg2rad(17.5)
window_thre = np.arctan(1.44/1.7)
baffle_thre = np.arctan(2.6/1.04)

axis_height = 61+152
window_height = axis_height+36+4+76
baffle_height = window_height+50+120

axis = np.array([0, 0, axis_height])
beam_v = np.array([0, 0, window_height - 70/2/np.tan(beam_thre)])
window_v = np.array([0, 0, window_height + 80/2/np.tan(window_thre)])
baffle_v = np.array([0, 0, baffle_height - 208/2/np.tan(baffle_thre)])

#colors = ['#7fffd4', '#bcecac', 'sandybrown', 'indianred']
#colors = ['tab:cyan', 'tab:green', 'tab:orange', 'red']
colors = ['dimgrey', 'forestgreen', 'darkgoldenrod', 'orangered']

def state(az, el, target):
    
    az, el = np.deg2rad(az-40), np.deg2rad(el)
    
    for point in target:
        if meas_angle(point, beam_v, az, el) < beam_thre:
            return 3, "< 17.5 deg", "hit FOV", colors[3]
    for point in target:
        if meas_angle(point, window_v, az, el) < window_thre:
            return 2, "< 41 deg", "hit Window", colors[2]
    for point in target:   
        if meas_angle(point, baffle_v, az, el) < baffle_thre:
            return 1, "< 68 deg", "hit Baffle", colors[1]       
    return 0, "> 68", "OK", colors[0]
    
def rotate_v(v, az, el):
    z_vector = v[2] - axis[2]
    offset = z_vector * np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    return axis + offset

def meas_angle(point, v, az, el):
    axis_vector = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    point_vector = point - rotate_v(v, az, el)
    return np.arccos(np.dot(axis_vector, point_vector) / (np.linalg.norm(axis_vector) * np.linalg.norm(point_vector)))

def return_color(angle):
    if angle < np.rad2deg(beam_thre):
        return colors[3]
    elif angle < np.rad2deg(window_thre):
        return colors[2]
    elif angle < np.rad2deg(baffle_thre):
        return colors[1]
    else:
        return colors[0]
    