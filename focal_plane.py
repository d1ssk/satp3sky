import numpy as np
import so3g.proj as proj
import sotodlib.sim_hardware as sh
import math
import quaternion
import matplotlib.pyplot as plt

class FocalPlane:
    # Get default focal plane from sotodlib
    def __init__(self, Az, El, Bs, use_saved=True):
        self.Az = Az
        self.El = El
        self.Bs = Bs
        
        if use_saved:
            dets_array = np.load('./data/dets.npy')
            dets = np.array([np.quaternion(*q) for q in dets_array], dtype=np.quaternion)
            centers_array = np.load('./data/centers.npy')
            centers = np.array([np.quaternion(*q) for q in centers_array], dtype=np.quaternion)
        else:
            dets = self.get_fp(save=True)
            centers = self.get_centers(save=True)
            
        dets_rot = self.rotate_point(dets, Az, El, Bs)
        centers_rot = self.rotate_point(centers, Az, El, Bs)
            
        dets_azel = np.array([self.qt2azel(q) for q in dets_rot])
        centers_azel = np.array([self.qt2azel(q) for q in centers_rot])
        
        self.dets_az = dets_azel[:,0]
        self.dets_el = dets_azel[:,1]
        self.centers_az = centers_azel[:,0]
        self.centers_el = centers_azel[:,1]
        
    def get_fp(self, save=False):
        xi_hw, eta_hw, dets_hw = self.get_hw_positions('f090')

        theta = np.arcsin(np.sqrt(xi_hw**2 + eta_hw**2))
        phi = np.arctan2(-xi_hw, -eta_hw)

        # Cartesian coordinates conversion
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        dets = np.array([np.quaternion(0, x, y, z) for x,y,z in zip(x.flatten(), y.flatten(), z.flatten())], dtype=np.quaternion)
        
        if save:
            dets_array = np.array([[det.w, det.x, det.y, det.z] for det in dets])
            np.save('./data/dets.npy', dets_array)
        
        return dets
    
    def get_centers(self, save=False):
        # 7 wafers
        wafers = ['w25', 'w26', 'w27', 'w28', 'w29', 'w30', 'w31']
        centers = []
        for wafer in wafers:
            xi_hw, eta_hw, dets_hw = self.get_hw_positions(wafer)
            xi_avg, eta_avg = np.mean(xi_hw), np.mean(eta_hw)
            theta = np.arcsin(np.sqrt(xi_avg**2 + eta_avg**2))
            phi = np.arctan2(-xi_avg, -eta_avg)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            centers.append(np.quaternion(0, x, y, z))
            
        centers = np.array(centers, dtype=np.quaternion)
            
        if save:
            centers_array = np.array([[center.w, center.x, center.y, center.z] for center in centers])
            np.save('./data/centers.npy', centers_array)
            
        return centers
        
    def rotate_point(self, q_p, az, el, bs):

        r_bs = np.deg2rad(bs)
        r_theta = np.deg2rad(90-el)
        r_phi = np.deg2rad(-az)

        q_bs = np.quaternion(np.cos(r_bs/2), 0, 0, np.sin(r_bs/2))
        q_theta = np.quaternion(np.cos(r_theta/2), 0, np.sin(r_theta/2), 0)
        q_phi = np.quaternion(np.cos(r_phi/2), 0, 0, np.sin(r_phi/2))

        q_p_rot = q_phi * q_theta * q_bs * q_p * q_bs.conjugate() * q_theta.conjugate() * q_phi.conjugate()
        
        return q_p_rot
    
    def qt2azel(self, q_p):

        x, y, z = q_p.x, q_p.y, q_p.z

        theta = np.arccos(z)
        phi = np.arctan2(y, x)

        az = -phi
        el = np.pi/2 - theta

        az, el = np.degrees(az)%360, np.degrees(el)
        
        return az, el
    
    def plot_fp(self):
        
        theta = np.deg2rad(90-self.dets_el)
        phi = np.deg2rad(-self.dets_az)
        # Cartesian coordinates conversion
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        # Creating a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting the points
        ax.scatter(x, y, z, color='r')

        # Creating a unit sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        # Plotting the unit sphere
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.3)

        # Setting the labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Showing the plot
        plt.show()
    
    @staticmethod
    def get_hw_positions(band):
        hw = sh.sim_nominal()
        sh.sim_detectors_toast(hw, 'SAT1')
        # Store specific detector quaternions (dichroic detectors)
        qdr, names_all = [], []
        for names in hw.data['detectors'].keys():
            if band in names:
                qdr.append([hw.data['detectors'][names]['quat'][3]] + list(hw.data['detectors'][names]['quat'][:3]))
                names_all.append(names)
        # Construct an so3g object from those
        quat_det = proj.quat.G3VectorQuat(np.array(qdr))
        # in radians
        xi_h,eta_h,gamma_h = proj.quat.decompose_xieta(quat_det)
        return xi_h, eta_h, names_all
