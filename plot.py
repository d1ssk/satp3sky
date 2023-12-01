import numpy as np
import datetime as dt
from zoneinfo import ZoneInfo
import so3g.proj as proj
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import imageio

import structures as st
from planets import Planets
import focal_plane as fp
import sun
import contamination as cont

legend_order = ['Sun', 'Moon', 'Jupiter', 'Saturn', 'tauA', 'GalCenter', 'Galaxy', "Horizon", 'Gantry', 'GS']
colors = {'Sun': 'tab:red', 'Moon': 'tab:purple', 'Jupiter': 'tab:green', 'Saturn': 'tab:brown', 'GalCenter': 'tab:grey', 'tauA': 'tab:cyan', 'Galaxy': 'tab:grey', "Horizon": 'black', "Gantry": 'tab:blue', "GS": 'tab:orange'}
sizes = {'Sun': 30, 'Moon': 30, 'Jupiter': 30, 'Saturn': 30, 'GalCenter': 30, 'tauA': 30, 'Galaxy': 1, "Horizon": 2.5, "Gantry": 3, "GS": 1.5}
obj_labels1 = {'Sun': 'Sun', 'Moon': 'Moon', 'Jupiter': 'Jupiter', 'Saturn': 'Saturn', 'GalCenter': 'Galactic Center', 'tauA': 'tau A', 'Galaxy': 'Galactic plane\n(GalCen or tauA give time)', "Horizon": 'Horizon', "Gantry": 'Gantry', "GS": 'Ground Shield'}
obj_labels2 = {'Sun': 'Sun', 'Moon': 'Moon', 'Jupiter': 'Jupiter', 'Saturn': 'Saturn', 'GalCenter': 'Galactic Center', 'tauA': 'tau A', 'Galaxy': 'Galactic plane', "Horizon": 'Horizon', "Gantry": 'Gantry', "GS": 'Ground Shield'}
markers = {'Sun': 'o', 'Moon': 'o', 'Jupiter': 'o', 'Saturn': 'o', 'GalCenter': 'o', 'tauA': 'o', 'Galaxy': '_', "Horizon": '_', "Gantry": 'o', "GS": '_'}
wafers = ['Mv5', 'Mv35', 'Mv27', 'Mv17', 'Mv33', 'Mv23', 'Mv12']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

so3gsite = proj.coords.SITES['so']
site = so3gsite.ephem_observer()
CHILE = ZoneInfo("America/Santiago")

ymin, ymax = 0, 90
xticks1 = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
xtickslabel1 = [280, 300, 320, 340, 0, 20, 40, 60, 80]

def sky_1day(date, site=site, zone=CHILE, focalplane=False, Az=0, El=90, Bs=0, r=0, save=False, filename="", exclude=[], plot_gal=True):
    """
    plot the sky every 1 hour for a given date
    """
    if type(date) != dt.datetime:
        date = date.split('/')
        start_time = dt.datetime(int(date[0]), int(date[1]), int(date[2]), 0, 0, tzinfo=zone)
    else:
        start_time = date
    delta = dt.timedelta(minutes=60)
    
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
    ax1.set_xlim(-90, 90)
    ax1.set_xticks(xticks1)
    ax1.set_xticklabels(xtickslabel1)
    ax2.set_xlim(90, 270)
    
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    plot_structures(ax1, ax2, r=r, exclude=exclude)
    
    current_time = start_time
    obj_dict = {}
    for planet in legend_order[:-3]:
        obj_dict[planet] = [[], []]
    for _ in range(24):
        pls = Planets(current_time, site, zone)
        for planet in pls.planets:
            obj_dict[planet][0].append(eval(f"pls.{planet}.az"))
            obj_dict[planet][1].append(eval(f"pls.{planet}.el"))
        current_time += delta
    
    hours = np.arange(0, 24, 1)
    plots_obj = set(pls.planets) - set(exclude)
    for obj in plots_obj:
        if obj != 'Galaxy':
            pass
            az = np.array(obj_dict[obj][0])
            el = np.array(obj_dict[obj][1])
            split_scatter(ax1, ax2, az, el, colors[obj], obj_labels1[obj], sizes[obj])
            split_label(ax1, ax2, az, el, colors[obj], hours, 8)
        else:
            if plot_gal:
                # Generate 24 different colored grey shades
                colored_greys = generate_colored_greys(24)
                for i in range(len(obj_dict[obj][0])):
                    az = np.array(obj_dict[obj][0][i])
                    el = np.array(obj_dict[obj][1][i])
                    split_plot(ax1, ax2, az, el, colored_greys[i], obj_labels1[obj], sizes[obj]/1.5, linestyles[i % 4], 0.6)
            
    if focalplane:
        plot_focalplane(ax1, ax2, Az, El, Bs)
        plot_fp_label(ax1, ax2, Az, El, Bs)
        
    fig.suptitle(start_time.strftime("%Y/%m/%d") + " (Time:Local)", fontsize=16)
    ax1.set_title('North', fontsize=12)
    ax2.set_title('South', fontsize=12)
    
    ax1.set_xlabel('Azimuth [deg]', fontsize=12)
    ax2.set_xlabel('Azimuth [deg]', fontsize=12)
    ax1.set_ylabel('Elevation [deg]', fontsize=12)
    ax2.set_ylabel('Elevation [deg]', fontsize=12)
    
    #fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    legend_list = set(pls.planets + ['Horizon', 'Gantry', 'GS']) - set(exclude)
    legend_elements = []
    for item in legend_order:
        if item in legend_list:
            # Check if the item is plotted as a line (e.g., 'Galaxy')
            if item == 'Galaxy' or item == 'Horizon' or item == 'GS':  # or other items plotted as lines
                # Create a line legend entry
                legend_elements.append(ml.lines.Line2D([0], [0], color=colors[item], linewidth=sizes[item], label=obj_labels1[item]))
            else:
                # Adjust marker size for the legend based on the actual size used in plot
                legend_marker_size = np.sqrt(sizes[item])
                # Create a scatter legend entry for other objects
                legend_elements.append(ml.lines.Line2D([0], [0], color=colors[item], marker='o', linestyle='None', markersize=legend_marker_size, label=obj_labels1[item]))

    # Add legend to the figure with custom elements
        
    if "Gantry" not in exclude:
        if r == 0:
            target = np.load("./data/Gantry.npy")
        else:
            target = st.Gantry(r=r, mode='optics').points
        num, text1, text2, color_g = cont.state(Az, El, target)
        legend_elements.append(ml.lines.Line2D([0], [0], marker='None', linestyle='None', label="Gantry: " + text2))
        
    legend = fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    if "Gantry" not in exclude:
        legend.get_texts()[-1].set_color(color_g)       
    
    fig.tight_layout()
        
    if save:
        plt.savefig("./img/" + filename, bbox_inches='tight')

def plot_sky(datetime, site=site, zone=CHILE, focalplane=False, Az=0, El=90, Bs=0, r=0, save=False, filename="", exclude=[]):
    """
    Plot the sky at a given datetime.
    """
    if type(datetime) != dt.datetime:
        datetime_list = datetime.split('/')
        datetime = dt.datetime(int(datetime_list[0]), int(datetime_list[1]), int(datetime_list[2]), int(datetime_list[3]), int(datetime_list[4]), tzinfo=zone)
    
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
    ax1.set_xlim(-90, 90)
    ax1.set_xticks(xticks1)
    ax1.set_xticklabels(xtickslabel1)
    ax2.set_xlim(90, 270)
    
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    
    plot_structures(ax1, ax2, r=r, exclude=exclude)
    
    pls = Planets(datetime, site, zone)
    plots_obj = set(pls.planets) - set(exclude)

    for obj in plots_obj:
        if obj != 'Galaxy':
            az = np.array([eval(f"pls.{obj}.az")])
            el = np.array([eval(f"pls.{obj}.el")])
            split_scatter(ax1, ax2, az, el, colors[obj], obj_labels2[obj], sizes[obj])
        else:
            az = eval(f"pls.{obj}.az")
            el = eval(f"pls.{obj}.el")
            split_plot(ax1, ax2, az, el, colors[obj], obj_labels2[obj], sizes[obj])
            
    if focalplane:
        plot_focalplane(ax1, ax2, Az, El, Bs)
        plot_fp_label(ax1, ax2, Az, El, Bs)
        
    fig.suptitle(datetime.strftime("%Y/%m/%d %H:%M") + " (Local)", fontsize=16)
    ax1.set_title('North', fontsize=12)
    ax2.set_title('South', fontsize=12)
    
    ax1.set_xlabel('Azimuth [deg]', fontsize=12)
    ax2.set_xlabel('Azimuth [deg]', fontsize=12)
    ax1.set_ylabel('Elevation [deg]', fontsize=12)
    ax2.set_ylabel('Elevation [deg]', fontsize=12)
    
    #fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    legend_list = set(pls.planets + ['Horizon', 'Gantry', 'GS']) - set(exclude)
    legend_elements = []
    for item in legend_order:
        if item in legend_list:
            # Check if the item is plotted as a line (e.g., 'Galaxy')
            if item == 'Galaxy' or item == 'Horizon' or item == 'GS':  # or other items plotted as lines
                # Create a line legend entry
                legend_elements.append(ml.lines.Line2D([0], [0], color=colors[item], linewidth=sizes[item], label=obj_labels2[item]))
            else:
                # Adjust marker size for the legend based on the actual size used in plot
                legend_marker_size = np.sqrt(sizes[item])
                # Create a scatter legend entry for other objects
                legend_elements.append(ml.lines.Line2D([0], [0], color=colors[item], marker='o', linestyle='None', markersize=legend_marker_size, label=obj_labels2[item]))
    sun_angle = sun.sun_angle(datetime, Az, El)
    color_sun = cont.return_color(sun_angle)
    if focalplane:
        legend_elements.append(ml.lines.Line2D([0], [0], marker='None', linestyle='None', label="Sun angle: " + f"{sun_angle:.0f} deg"))

    # Add legend to the figure with custom elements
        
    if "Gantry" not in exclude:
        if r == 0:
            target = np.load("./data/Gantry.npy")
        else:
            target = st.Gantry(r=r, mode='optics').points
        num, text1, text2, color_g = cont.state(Az, El, target)
        legend_elements.append(ml.lines.Line2D([0], [0], marker='None', linestyle='None', label="Gantry: " + text2))
        
    legend = fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    if focalplane and "Gantry" not in exclude:
        legend.get_texts()[-2].set_color(color_sun)
        legend.get_texts()[-1].set_color(color_g)
    elif focalplane and "Gantry" in exclude:
        legend.get_texts()[-1].set_color(color_sun)
        
    
    fig.tight_layout()
        
    if save:
        plt.savefig("./img/" + filename, bbox_inches='tight')
        
def plot_timestream(start, end, delta, site=site, zone=CHILE, focalplane=False, Az=0, El=90, Bs=0, r=0, save=False, foldername="", filename="", exclude=[]):
    if type(start) != dt.datetime:
        start_list = start.split('/')
        start = dt.datetime(int(start_list[0]), int(start_list[1]), int(start_list[2]), int(start_list[3]), int(start_list[4]), tzinfo=zone)
    if type(end) != dt.datetime:
        end_list = end.split('/')
        end = dt.datetime(int(end_list[0]), int(end_list[1]), int(end_list[2]), int(end_list[3]), int(end_list[4]), tzinfo=zone)
    current_time = start
    delta = dt.timedelta(minutes=delta)
    i = 0
    while current_time <= end:
        plot_sky(current_time, site=site, zone=zone, focalplane=focalplane, Az=Az, El=El, Bs=Bs, r=r, save=save, filename=foldername + "/" + filename + f"_{i:03d}.png", exclude=exclude)
        current_time += delta
        i += 1
    if save:
        images = []
        fns = [f"./img/{foldername}/{filename}_{i:03d}.png" for i in range(i)]
        for fn in fns:
            images.append(imageio.imread(fn))
        imageio.mimsave('./img/' + foldername + '/' + filename + '.gif', images, duration=100, loop=0)
    
    
def plot_focalplane(ax1, ax2, Az, El, Bs):
    FP = fp.FocalPlane(Az, El, Bs)
    split_scatter(ax1, ax2, FP.dets_az, FP.dets_el, 'black', 'Focal Plane', 0.1, 0.1, 'x')
    
def plot_fp_label(ax1, ax2, Az, El, Bs):
    FP = fp.FocalPlane(Az, El, Bs)
    for i in range(FP.centers_az.size):
        if FP.centers_az[i] > 270:
            ax1.text(FP.centers_az[i]-360, FP.centers_el[i], wafers[i], fontsize=8)
        else:
            if FP.centers_az[i] > 90:
                ax2.text(FP.centers_az[i]-2, FP.centers_el[i], wafers[i], fontsize=8)
            else:
                ax1.text(FP.centers_az[i]-2, FP.centers_el[i], wafers[i], fontsize=8)
    
def plot_structures(ax1, ax2, r=0, exclude=[]):
    
    if "Horizon" not in exclude:
        az = st.Horizon_az
        el = st.Horizon_el
        split_plot(ax1, ax2, az, el, colors['Horizon'], 'Horizon', sizes['Horizon'])
    
    if "Gantry" not in exclude:
        if r != 0:
            gt = st.Gantry(r=r).angles
        else:
            gt = np.load('./data/Gantry_plot.npy')
        az, el = gt[:,0], gt[:,1]
        az1, el1, az2, el2 = split_azel(az, el)
        ax1.scatter(az1, el1, s=3, color=colors['Gantry'], label='Gantry')
        ax2.scatter(az2, el2, s=3, color=colors['Gantry'])
        
    if "GS" not in exclude:
        gs = np.load('./data/GS_plot.npy')
        ax1.axhline(gs[0], color=colors['GS'], label='Ground Shield')
        ax1.axhline(gs[1], color=colors['GS'])
        ax1.axhspan(0, gs[1], color=colors['GS'], alpha=0.2)
        ax2.axhline(gs[0], color=colors['GS'])
        ax2.axhline(gs[1], color=colors['GS'])
        ax2.axhspan(0, gs[1], color=colors['GS'], alpha=0.2)

def split_azel(az, el):
    iwh = np.where(az > 270)
    az[iwh] = az[iwh] - 360
    iwh1 = np.where(np.logical_and(az >= -90., az <= 90.))
    iwh2 = np.where(np.logical_and(az >= 90., az <= 270.))
    return az[iwh1], el[iwh1], az[iwh2], el[iwh2]

def split_scatter(ax1, ax2, az, el, color, label, size, alpha=1, marker='o'):
    az1, el1, az2, el2 = split_azel(az, el)
    ax1.scatter(az1, el1, s=size, color=color, label=label, alpha=alpha ,marker=marker)
    ax2.scatter(az2, el2, s=size, color=color, alpha=alpha ,marker=marker)
    
def split_plot(ax1, ax2, az, el, color, label, size, linetype='solid', alpha=1):
    az1, el1, az2, el2 = split_azel(az, el)
    ii1 = az1.argsort()
    ii2 = az2.argsort()
    ax1.plot(az1[ii1], el1[ii1], color=color, label=label, linewidth=size, alpha=alpha, linestyle=linetype)
    ax2.plot(az2[ii2], el2[ii2], color=color, linewidth=size, alpha=alpha, linestyle=linetype)

def split_label(ax1, ax2, az, el, color, labels, size, alpha=1):
    for i, label in enumerate(labels):
        if el[i] > 0:     
            if az[i] > 270:
                ax1.text(az[i]-361, el[i]+2, label, fontsize=size, color=color, alpha=alpha)
            elif az[i] > 90:
                ax2.text(az[i]-1, el[i]+2, label, fontsize=size, color=color, alpha=alpha)
            else:
                ax1.text(az[i]-1, el[i]+2, label, fontsize=size, color=color, alpha=alpha)
                
def generate_colored_greys(n):
    """Generate n different colored grey shades."""
    base_grey = 0.5  # Base value for grey (midpoint between black and white)
    colors = []
    for i in range(n):
        # Slightly alter the RGB values, keeping them close to the base grey
        r = base_grey + (i % 3 - 1) * 0.15 * (i / n)
        g = base_grey + ((i + 1) % 3 - 1) * 0.15 * (i / n)
        b = base_grey + ((i + 2) % 3 - 1) * 0.15 * (i / n)
        colors.append((r, g, b))
    return colors
