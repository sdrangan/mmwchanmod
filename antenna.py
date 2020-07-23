"""
antenna.py:  Classes for antenna modeling
"""
import numpy as np


    
class Elem3GPP(object):
    """
    Class for computing the element gain using 3GPP BS model
    """
    def __init__(self, phi0=0, theta0=0, phibw=120, thetabw=65):
        """
        Constructor

        Parameters
        ----------
        phi0, theta0 : scalars
            Center azimuth and elevation angle in degrees
        phibw, thetabw : scalars
            Azimuth and elevation half-power beamwidth in degrees
            A value <0 indicates that there is no directivity 
            in the azimuth or elevation directions

        """
        self.phi0 = phi0
        self.theta0 = theta0
        self.phibw = phibw
        self.thetabw = thetabw
        self.gain_max = 0
        
        # Other parameters
        self.slav = 30  # vertical side lobe
        self.Am = 30    # min gain
        
        # Calibrate
        self.calibrate()
        
    def response(self,phi,theta):
        """
        Computes antenna gain for angles
        
        Parameters
        ----------
        phi:  array
            Azimuth angles in degrees
        theta: array
            Elevation angles in degrees
        """
        if self.thetabw > 0:
            Av = -np.minimum( 12*((theta-self.theta0)/self.thetabw)**2, self.slav)
        else:
            Av = 0
        if self.phibw > 0:
            Ah = -np.minimum( 12*((phi-self.phi0)/self.phibw)**2, self.Am)
        else:
            Ah = 0
        gain = self.gain_max - np.minimum(-Av-Ah, self.Am)
        return gain
        
    
    def calibrate(self, ns=10000):
        """
        Calibrates the maximum antenna gain

        Parameters
        ----------
        ns : int
            Number of samples used in the calibration
        """
        
        # Generate random angles
        phi = np.random.uniform(-180,180,ns)
        theta = np.random.uniform(0,180,ns)
        
        # Compute weigths
        w = np.sin(theta*np.pi/180)
        
        # Get gain in linear scale
        gain = self.response(phi, theta)
        gain_lin = 10**(0.1*gain)
        
        # Find the mean gain 
        gain_mean = np.mean(gain_lin*w)/ np.mean(w)
        gain_mean = 10*np.log10(gain_mean)
        
        # Adjust the max gain
        self.gain_max = self.gain_max - gain_mean
        
class Elem3GPPMultiSector(object):
    """
    Element gain for a multi-sector site .
    
    Finds the max gain along multiple sectors arranged along
    the azimuth plane
    """   
    def __init__(self, nsect=3, phi0=0, theta0=0, phibw=65, thetabw=65):
        """
        Constructor

        Parameters
        ----------
        nsect : int
            Number of sectors
        phi0, theta0 : scalars
            Center azimuth and elevation angle in degrees.
            The value `phi0` is the azimuth angle for sector 0.
        phibw, thetabw : scalars
            Azimuth and elevation half-power beamwidth in degrees

        """     
        
        # Create the elements
        self.elem = []
        for i in range(nsect):
            # Angle of the sector
            phi_sect = phi0 + i*360/(nsect+1)
            elem_sect = Elem3GPP(phi_sect,theta0,phibw,thetabw)
            self.elem.append(elem_sect)
            
    def response(self,phi,theta):
        """
        Computes antenna gain for angles.  This is found from the
        maximum gain across the sectors
        
        Parameters
        ----------
        phi:  array
            Azimuth angles in degrees
        theta: array
            Elevation angles in degrees
        """        
        nsect = len(self.elem)
        for i in range(nsect):
            gaini = self.elem[0].response(phi, theta)
            if i == 0:
                gain = gaini
            else:
                gain = np.maximum(gain, gaini)
        return gain
      
        

        
        
        
        

