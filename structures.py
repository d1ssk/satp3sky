import numpy as np
from matplotlib import pyplot as plt

# The axis of the platform
axis_height = 61+152
axis = np.array([0, 0, axis_height])

class Gantry:
    """
    A class to represent a gantry. Unit of length is cm.
    """
    
    def __init__(self, r=0, mode='plot'):
        self.r = r
        self.get_geometry()
        self.get_points(mode=mode)
        self.get_angles()
        
    def get_geometry(self):
        
        r = self.r
        
        self.bar_bottom_height = 671
        self.bar_length = 70
        
        self.xc_bar = 0
        self.yc_bar = (536*(1-r)+(-22)*r)-112
        self.zc_bar = self.bar_bottom_height+self.bar_length/2

        self.xl_bar = 750
        self.yl_bar = 20
        self.zl_bar = self.bar_length

        self.xc_lleg = -747/2
        self.yc_lleg = (536*(1-r)+(-22)*r)-112
        self.zc_lleg = 0

        self.xl_lleg = 50
        self.yl_lleg = 224
        self.zl_lleg = 700

        self.xc_rleg = 747/2
        self.yc_rleg = (536*(1-r)+(-22)*r)-112
        self.zc_rleg = 0

        self.xl_rleg = 50
        self.yl_rleg = 224
        self.zl_rleg = 700
        
    def get_points(self, mode='plot', cut_bar=None, cut_lleg=None, cut_rleg=None):
        """
        Return the points of the gantry.
        """
        
        if mode == 'plot':
            cut_bar = (30, 10, 10)
            cut_lleg = (10, 10, 20)
            cut_rleg = (10, 10, 20)
        elif mode == 'optics':
            cut_bar = (10, 2, 4)
            cut_lleg = (2, 2, 4)
            cut_rleg = (2, 2, 4)
        elif mode == 'arb':
            pass
        
        bar = self.generate_cuboid_points((self.xc_bar, self.yc_bar, self.zc_bar), (self.xl_bar, self.yl_bar, self.zl_bar), cut_bar)
        lleg = self.generate_triangle_with_height_points((self.xc_lleg, self.yc_lleg, self.zc_lleg), self.xl_lleg, self.yl_lleg, self.zl_lleg, cut_lleg)
        rleg = self.generate_triangle_with_height_points((self.xc_rleg, self.yc_rleg, self.zc_rleg), self.xl_rleg, self.yl_rleg, self.zl_rleg, cut_rleg)
        
        if mode == 'optics':
            
            u = 3
            r = self.r

            xc_lleg_upper = -747/2
            yc_lleg_upper = (536*(1-r)+(-22)*r)-112
            zc_lleg_upper = self.zl_lleg*(u-1)/u

            xl_lleg_upper = 50
            yl_lleg_upper = 224/u
            zl_lleg_upper = 700/u

            lleg_upper = self.generate_triangle_with_height_points((xc_lleg_upper, yc_lleg_upper, zc_lleg_upper), xl_lleg_upper, yl_lleg_upper, zl_lleg_upper, (2, 2, 5))
            lleg += lleg_upper
            
            xc_rleg_upper = 747/2   
            yc_rleg_upper = (536*(1-r)+(-22)*r)-112
            zc_rleg_upper = self.zl_rleg*(u-1)/u

            xl_rleg_upper = 50
            yl_rleg_upper = 224/u
            zl_rleg_upper = 700/u

            rleg_upper = self.generate_triangle_with_height_points((xc_rleg_upper, yc_rleg_upper, zc_rleg_upper), xl_rleg_upper, yl_rleg_upper, zl_rleg_upper, (2, 2, 5))
            rleg += rleg_upper
        
        gantry = bar + lleg + rleg
        self.points = gantry
        
    def get_angles(self):
        """
        Return the angles of the gantry.
        """
        self.angles = np.array([point_angle(point, axis) for point in self.points])
        
    def save_points(self, filename):
        """
        Save the points of the gantry to a file.
        """
        np.save("./data/" + filename + ".npy", self.points)
        
    def save_angles(self, filename):
        """
        Save the angles of the gantry to a file.
        """
        np.save("./data/" + filename + ".npy", self.angles)
    
    @staticmethod
    def generate_cuboid_points(center, lengths, cuts):
        # Unpack the center and lengths
        cx, cy, cz = center
        lx, ly, lz = lengths
        cut_x, cut_y, cut_z = cuts

        # Calculate half lengths
        half_lx, half_ly, half_lz = lx / 2, ly / 2, lz / 2

        # Initialize a set for points to avoid duplicates
        points = set()

        # Function to add points on a face
        def add_face_points_z(x_range, y_range, z):
            for x in x_range:
                for y in y_range:
                    points.add((x, y, z))
                    
        def add_face_points_y(x_range, y, z_range):
            for x in x_range:
                for z in z_range:
                    points.add((x, y, z))
        
        def add_face_points_x(x, y_range, z_range):
            for y in y_range:
                for z in z_range:
                    points.add((x, y, z))

        # Generate points on each face
        # Front and Back faces
        x_range = [cx - half_lx + i * lx / cut_x for i in range(cut_x + 1)]
        y_range = [cy - half_ly + i * ly / cut_y for i in range(cut_y + 1)]
        add_face_points_z(x_range, y_range, cz - half_lz)
        add_face_points_z(x_range, y_range, cz + half_lz)

        # Left and Right faces
        z_range = [cz - half_lz + i * lz / cut_z for i in range(cut_z + 1)]
        x_range = [cx - half_lx + i * lx / cut_x for i in range(cut_x + 1)]
        add_face_points_y(x_range, cy - half_ly, z_range)
        add_face_points_y(x_range, cy + half_ly, z_range)
        
        # Top and Bottom faces
        y_range = [cy - half_ly + i * ly / cut_y for i in range(cut_y + 1)]
        z_range = [cz - half_lz + i * lz / cut_z for i in range(cut_z + 1)]
        add_face_points_x(cx - half_lx, y_range, z_range)
        add_face_points_x(cx + half_lx, y_range, z_range)

        return [np.array(point) for point in points]    
    
    @staticmethod
    def generate_triangle_with_height_points(center, bottom_x, bottom_y, height, cuts):
        # Unpack the center
        cx, cy, cz = center
        cut_bottom_x, cut_bottom_y, cut_height = cuts

        # Calculate vertices of the triangular base
        v1 = (cx - bottom_x / 2, cy - bottom_y / 2, cz)
        v2 = (cx - bottom_x / 2, cy + bottom_y / 2, cz)
        v3 = (cx - bottom_x / 2, cy, cz + height)
        
        v4 = (cx + bottom_x / 2, cy - bottom_y / 2, cz)
        v5 = (cx + bottom_x / 2, cy + bottom_y / 2, cz)
        v6 = (cx + bottom_x / 2, cy, cz + height)

        # Initialize a set for points to avoid duplicates
        points = set()

        # Function to add points on a triangle face
        def add_triangle_face_points(v1, v2, v3, cuts, x):
            for i in range(cuts + 1):
                for j in range(cuts - i + 1):
                    alpha = i / cuts
                    beta = j / cuts
                    gamma = 1 - alpha - beta
                    y = alpha * v1[1] + beta * v2[1] + gamma * v3[1]
                    z = alpha * v1[2] + beta * v2[2] + gamma * v3[2]
                    points.add((x, y, z))

        # Add points on the triangular base and top
        add_triangle_face_points(v1, v2, v3, cut_height, cx - bottom_x / 2)
        add_triangle_face_points(v1, v2, v3, cut_height, cx + bottom_x / 2)

        # Function to add points on a bottom rectanguler
        def add_bottom_rectangular_face_points(cut_bottom_x, cut_bottom_y):
            for i in range(cut_bottom_x + 1):
                for j in range(cut_bottom_y + 1):
                    x = v1[0] + i * (v4[0] - v1[0]) / cut_bottom_x
                    y = v1[1] + j * (v2[1] - v1[1]) / cut_bottom_y
                    z = cz
                    points.add((x, y, z))
                    
        # Add points on the bottom rectangular faces
        add_bottom_rectangular_face_points(cut_bottom_x, cut_bottom_y)
        
        # Function to add points on side rectangulers
        def add_side_rectangular_face_points(v1, v2, v3, v4, cut_height, cut_bottom):
            for i in range(cut_height + 1):
                for j in range(cut_bottom + 1):
                    x = v1[0] + j * (v2[0] - v1[0]) / cut_bottom
                    y = v1[1] + i * (v3[1] - v1[1]) / cut_height
                    z = v1[2] + i * (v4[2] - v1[2]) / cut_height
                    points.add((x, y, z))
                    
        # Add points on the side rectangular faces
        add_side_rectangular_face_points(v1, v4, v3, v6, cut_height, cut_bottom_x)
        add_side_rectangular_face_points(v2, v5, v3, v6, cut_height, cut_bottom_x)
        
        return [np.array(point) for point in points]

    
class GroundScreen:
    """
    A class to represent a ground screen. Unit of length is cm.
    """
    
    def __init__(self, mode='plot'):
        self.get_geometry()
        self.get_points(mode=mode)
        self.get_angles()
        
    def get_geometry(self):
        self.bottom_rad = 1400/2
        self.bottom_height = 363
        self.top_rad = 1680/2
        self.top_height = 200
        self.total_height = self.bottom_height + self.top_height/np.sqrt(2)
        self.top_med_height = self.bottom_height + self.top_height/2/np.sqrt(2)
        self.top_med_rad = self.bottom_rad + self.top_height/2/np.sqrt(2)
        
    def get_points(self, mode='plot'):
        
        if mode == 'plot':
            gs = self.generate_cylindrical_points((0, 0, self.bottom_height), self.bottom_rad, 1, 30, 1)
            gs += self.generate_cylindrical_points((0, 0, self.total_height), self.top_rad, 1, 30, 1)
        elif mode == 'optics':
            gs = self.generate_cylindrical_points((0, 0, self.bottom_height/2), self.bottom_rad, self.bottom_height, 30, 3)
            gs += self.generate_cylindrical_points((0, 0, self.top_med_height), self.top_med_rad, 1, 30, 1)
            gs += self.generate_cylindrical_points((0, 0, self.total_height), self.top_rad, 1, 30, 1)
        self.points = gs
            
    def get_angles(self):
        """
        Return the angles of the ground screen.
        """
        gs_angles = np.array([point_angle(self.points[0], axis)[1], point_angle(self.points[-1], axis)[1]])
        self.angles = gs_angles
        
    def save_points(self, filename):
        """
        Save the points of the ground screen to a file.
        """
        np.save("./data/" + filename + ".npy", self.points)
        
    def save_angles(self, filename):
        """
        Save the angles of the ground screen to a file.
        """
        np.save("./data/" + filename + ".npy", self.angles)
        
    @staticmethod
    def generate_cylindrical_points(center, radius, height, cut_radius, cut_height):
        points = set()

        # Function to add points on a face
        def add_face_points_circle(h):
            for theta in np.linspace(0, 2 * np.pi, cut_radius + 1):
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                points.add((x, y, h))
                
        for h in np.linspace(center[2] - height / 2, center[2] + height / 2, cut_height + 1):
            add_face_points_circle(h)
            
        return [np.array(point) for point in points]
    
def point_angle(point, axis):
    
    distance = np.linalg.norm(point-axis)
    el = np.rad2deg(np.arcsin((point[2]-axis[2])/distance))
    az = np.rad2deg(np.arctan2(point[0]-axis[0], point[1]-axis[1]))+40
    return az, el

def plot_points(points_list):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for points in points_list:
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        z_coords = [point[2] for point in points]
        ax.scatter(x_coords, y_coords, z_coords)

    # equalize the scales of the axes
    ax.set_aspect('equal')
    #plt.savefig('/so/home/dsasaki/planet/img/gantry_gs.pdf')
    plt.show()
    
Horizon_az = np.arange(0, 360, 6)
Horizon_el = 0.478*np.array([0.99,
 4.07,
 11.19,
 15.9,
 21.0,
 26.91,
 31.53,
 31.1,
 24.21,
 20.63,
 25.69,
 25.31,
 18.49,
 15.55,
 14.22,
 12.32,
 11.27,
 10.58,
 7.19,
 7.17,
 10.1,
 10.1,
 7.73,
 3.99,
 3.09,
 3.48,
 4.08,
 4.39,
 4.63,
 1.77,
 0.94,
 0,
 0,
 0,
 0,
 0,
 0,
 0,
 -0.94,
 -0.71,
 -1.05,
 -0.89,
 -0.93,
 -1.26,
 -1.75,
 -2.96,
 -3.5,
 -4.19,
 -4.26,
 -4.26,
 -4.14,
 -2.8,
 -4.07,
 4.07,
 2.51,
 4.59,
 1.17,
 3.81,
 3.12,
 0])