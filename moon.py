import numpy as np
import datetime as dt
import numpy as np
from zoneinfo import ZoneInfo
import so3g.proj as proj
import ephem
from matplotlib import pyplot as plt

so3gsite = proj.coords.SITES['so']
site = so3gsite.ephem_observer()
CHILE = ZoneInfo("America/Santiago")

def moon_angle(datetime, az, el, site=site, zone=CHILE):
    
    if type(datetime) != dt.datetime:
        datetime_list = datetime.split('/')
        datetime = dt.datetime(int(datetime_list[0]), int(datetime_list[1]), int(datetime_list[2]), int(datetime_list[3]), int(datetime_list[4]), tzinfo=zone)
        
    site.date = ephem.Date(datetime)
    moon = ephem.Moon(site)
    
    az = np.deg2rad(az)
    el = np.deg2rad(el)
    
    az_moon = moon.az
    el_moon = moon.alt
    
    angle = meas_angle(az, el, az_moon, el_moon)
    
    return angle

def moon_angles(start, end, delta, Az, El, site=site, zone=CHILE):

    if type(start) != dt.datetime:
        start_list = start.split('/')
        start = dt.datetime(int(start_list[0]), int(start_list[1]), int(start_list[2]), int(start_list[3]), int(start_list[4]), tzinfo=zone)
        
    if type(end) != dt.datetime:
        end_list = end.split('/')
        end = dt.datetime(int(end_list[0]), int(end_list[1]), int(end_list[2]), int(end_list[3]), int(end_list[4]), tzinfo=zone)

    delta = dt.timedelta(minutes=delta)
    
    az = np.deg2rad(Az)
    el = np.deg2rad(El)
    
    data = []
    i = 0

    # Calculate moon position every 10 minutes
    current_time = start
    while current_time <= end:
        site.date = ephem.Date(current_time)
        current_time += delta

        #site.date = ephem.Date( dt.datetime.now())

        moon = ephem.Moon(site)

        az_moon = moon.az
        el_moon = moon.alt
        
        angle = meas_angle(az, el, az_moon, el_moon)
        data.append([current_time, angle])
    
    return np.array(data)

def plot_moon_angles(Az, El, start, end, delta, thre=45, site=site, zone=CHILE):
    
    data = moon_angles(start, end, delta, Az, El, site=site, zone=zone)
    datatime = [data[i][0] for i in range(len(data))]
    angle = [data[i][1] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(9, 5))  # Adjust as needed
    ax.plot(datatime, angle)
    ax.axhline(y=thre, color='r', linestyle='-')

    # cross point of the line and the curve
    cp = []
    for i in range(len(datatime)-1):
        if angle[i] < thre and angle[i+1] > thre:
            cp.append(datatime[i])
        elif angle[i] > thre and angle[i+1] < thre:
            cp.append(datatime[i])
            
    # plot the cross point
    for i in range(len(cp)):
        ax.axvline(x=cp[i], color='black', linestyle='--')
        
    ax.set_ylabel('Moon angle [deg]')
    ax.set_xlabel('Time (Local)')
    plt.show()
        
    thres = []
    for item in cp:
        thres.append(item.strftime("%H:%M"))
        
    thres_txt = ", ".join(thres)
    
    print(f"{thre} deg threshold: " + thres_txt) 

def meas_angle(az1, el1, az2, el2):
    
    x1 = np.cos(el1)*np.sin(az1)
    y1 = np.cos(el1)*np.cos(az1)
    z1 = np.sin(el1)
    
    x2 = np.cos(el2)*np.sin(az2)
    y2 = np.cos(el2)*np.cos(az2)
    z2 = np.sin(el2)
    
    dot = x1*x2 + y1*y2 + z1*z2
    return np.rad2deg(np.arccos(dot))
    