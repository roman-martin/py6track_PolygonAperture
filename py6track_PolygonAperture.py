import numpy as np
from pysixtrack.elements import Element

class LimitPolygon(Element):
    # the input coords must be ordered, i.e the lines connecting 
    # neighbours must be the sides of the polygon
    #
    # convention: aperture[0] = x-coords, aperture[1] = y-coords
    _description = [
        (
            "aperture", 
            "m",
            "Array with coords of polygon vertices",
            lambda: [[-1.0,1.0,1.0,-1.0], [1.0,1.0,-1.0,-1.0]]
        )
    ]
      
    @staticmethod
    def is_right_of(line_start, line_end, point):
        # decides if refpoint is left of line going through line_start and 
        # line_end by checking the 3rd dimension direction of the cross 
        # product of the involved vectors
        def __crossproduct_z__(vector1, vector2):
            return np.multiply(vector1[0],vector2[1]) - np.multiply(vector1[1],vector2[0])
        
        return __crossproduct_z__(line_end-line_start, point-line_start) < 0.0
    

    def track(self, particle):
        aper_1 = np.expand_dims(self.aperture, axis=2)
        aper_2 = np.roll(aper_1, 1, axis=1)
        # prepare reference point outside aperture
        refpoint = 1.1*np.array([[abs(max(aper_1[0]))], [abs(max(aper_1[1]))]] )
        
        coords = np.array([[particle.x], [particle.y]])

        particle_is_right = self.is_right_of(aper_1, aper_2, coords)
        refpoint_is_right = self.is_right_of(aper_1, aper_2, refpoint)
        
        aper_1_is_right = self.is_right_of(coords, refpoint, aper_1)
        aper_2_is_right = self.is_right_of(coords, refpoint, aper_2)

        lines_intersect = np.where( np.logical_and(
                                       np.logical_xor(particle_is_right,refpoint_is_right),
                                       np.logical_xor(aper_1_is_right,aper_2_is_right)
                                    ),
                                    np.ones(aper_1[0].shape, dtype=np.int_),
                                    np.zeros(aper_1[0].shape, dtype=np.int_)
                                  ) # todo: make sure that end points are included
        num_intersects = np.sum(lines_intersect, axis=0)
        # if num_intersect is odd -> particle is inside aperture
        particle.state = np.where(  num_intersects % 2 == 1,
                                    np.ones(num_intersects.shape, dtype=np.int_),
                                    np.zeros(num_intersects.shape, dtype=np.int_)
                                 )
        particle.remove_lost_particles()
        if len(particle.state) == 0:
            return "All particles lost"


