import numpy as np
from zoneinfo import ZoneInfo
import so3g.proj as proj
import ephem
from astropy import units as u
from astropy.coordinates import SkyCoord

so3gsite = proj.coords.SITES['so']
site = so3gsite.ephem_observer()
CHILE = ZoneInfo("America/Santiago")

class Planets:
    
    def __init__(self, datetime, site=site, zone=CHILE):
        self.datetime = datetime
        site.date = ephem.Date(datetime)
        self.site = site
        self.zone = zone
        self.planets = ['Sun', 'Moon', 'Jupiter', 'Saturn', 'tauA', 'Galaxy', 'GalCenter']
        for planet in self.planets:
            exec(f"self.{planet} = Planet('{planet}', datetime, site, zone)")
        
class Planet:
    
    def __init__(self, name, datetime, site=site, zone=CHILE):
        
        self.name = name
        
        if name == 'GalCenter':
            obj = ephem.FixedBody()
            obj._ra = 17.7611*np.pi/12.
            obj._dec = -28.95*np.pi/180.
            obj.compute(site)
            
        elif name == 'tauA':
            obj = ephem.FixedBody()
            obj._ra = 5.5755*np.pi/12.
            obj._dec = 22.0167*np.pi/180.
            obj.compute(site)
            
        elif name == 'Galaxy':
            gl_v = np.linspace(0, 359.9, 72)
            naz = gl_v.size
            gb_v = np.zeros(naz)
            geq_gal = SkyCoord(l=gl_v*u.deg, b=gb_v*u.deg, frame= 'galactic')
            geq = geq_gal.transform_to('icrs')
            geq_ephem = [ephem.FixedBody() for i in range(naz)]
            for i in range(naz):
                geq_ephem[i]._ra = geq[i].ra.rad
                geq_ephem[i]._dec = geq[i].dec.rad
            geq_az = np.zeros([naz])
            geq_el = np.zeros([naz])
            for i in range(naz):
                geq_ephem[i].compute(site)
                geq_az[i] = np.rad2deg(geq_ephem[i].az)%360
                geq_el[i] = np.rad2deg(geq_ephem[i].alt)
            self.el = geq_el
            self.az = geq_az
            return
            
        else:
            obj = eval(f"ephem.{name}(site)")
            
        self.az = np.rad2deg(obj.az)%360    
        self.el = np.rad2deg(obj.alt)
