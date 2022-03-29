from math import sin, cos, sqrt, atan, atan2, degrees, radians
import math
import numpy as np
class Transformacje:
    def __init__(self, model: str = "wgs84"):
        """
        Parametry elipsoid:
            a - duża półoś elipsoidy - promień równikowy
            b - mała półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
        """
        if model == "wgs84":
            self.a = 6378137.0 # semimajor_axis
            self.b = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.a = 6378137.0
            self.b = 6356752.31414036
        elif model == "mars":
            self.a = 3396900.0
            self.b = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.a - self.b) / self.a
        self.ecc = sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 
        self.ecc2 = (2 * self.flat - self.flat ** 2) # eccentricity**2


    
    def xyz2plh(self, X, Y, Z, output = 'dec_degree'):
        """
        Algorytm Hirvonena - algorytm transformacji współrzędnych ortokartezjańskich (x, y, z)
        na współrzędne geodezyjne długość szerokość i wysokośc elipsoidalna (phi, lam, h). Jest to proces iteracyjny. 
        W wyniku 3-4-krotneej iteracji wyznaczenia wsp. phi można przeliczyć współrzędne z dokładnoscią ok 1 cm.     
        Parameters
        ----------
        X, Y, Z : FLOAT
             współrzędne w układzie orto-kartezjańskim, 

        Returns
        -------
        lat
            [stopnie dziesiętne] - szerokość geodezyjna
        lon
            [stopnie dziesiętne] - długośc geodezyjna.
        h : TYPE
            [metry] - wysokość elipsoidalna
        output [STR] - optional, defoulf 
            dec_degree - decimal degree
            dms - degree, minutes, sec
        """
        r   = sqrt(X**2 + Y**2)           # promień
        lat_prev = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przybliilizenie
        lat = 0
        while abs(lat_prev - lat) > 0.000001/206265:    
            lat_prev = lat
            N = self.a / sqrt(1 - self.ecc2 * sin(lat_prev)**2)
            h = r / cos(lat_prev) - N
            lat = atan((Z/r) * (((1 - self.ecc2 * N/(N + h))**(-1))))
        lon = atan(Y/X)
        N = self.a / sqrt(1 - self.ecc2 * (sin(lat))**2);
        h = r / cos(lat) - N       
        if output == "dec_degree":
            return degrees(lat), degrees(lon), h 
        elif output == "dms":
            lat = self.deg2dms(degrees(lat))
            lon = self.deg2dms(degrees(lon))
            return f"{lat[0]:02d}:{lat[1]:02d}:{lat[2]:.2f}", f"{lon[0]:02d}:{lon[1]:02d}:{lon[2]:.2f}", f"{h:.3f}"
        else:
            raise NotImplementedError(f"{output} - output format not defined")
            
    
            
    
            


    def s_A_z2neu (self , s, A, Zx):
        A = np.deg2rad(A)
        Zx = np.deg2rad(Zx)
        n = s * np.sin(Zx) * np.cos(A)
        e = s * np.sin(Zx) * np.sin(A)
        u = s * np.cos(Zx)
        
        return n, e, u



    def neu2dxyz(self , F1, L1, n, e, u):
        
        R = np.array([[-np.sin(F1) * np.cos(L1), -np.sin(L1), np.cos(F1) * np.cos(L1)],
                     [-np.sin(F1) * np.sin(L1), np.cos(L1), np.cos(F1) * np.sin(L1)],
                     [np.cos(F1), 0, np.sin(F1)]])
        dx = np.linalg.inv(R.transpose()) @ np.array([n, e, u])
        return dx
    
    
    
    
    
    def kivioji (self,fi_A, lam_A, s_ab, A_ab, a, e2):
        
        A_ab = np.deg2rad(A_ab)
        fi_A = np.deg2rad(fi_A)
        lam_A = np.deg2rad(lam_A)
        
        n = round(s_ab/1000)
        ds = s_ab/n
        
        for x in range(n):
            M = (a*(1-e2))/(math.sqrt(1-e2*(np.sin(fi_A))**2)**3)
            N = a/(math.sqrt(1-e2*(np.sin(fi_A))**2))
            dfi = ds*np.cos(A_ab)/M
            dA = ((np.sin(A_ab)*np.tan(fi_A))/N)*ds
            fim = fi_A + dfi/2
            Am = A_ab + dA/2
            Mm = (a*(1-e2))/(math.sqrt(1-e2*(np.sin(fim))**2)**3)
            Nm = a/(math.sqrt(1-e2*(np.sin(fim))**2))
            dfip = (ds*np.cos(Am))/Mm
            dlamp = (ds*np.sin(Am))/(Nm*np.cos(fim))
            dAp = ((np.sin(Am)*np.tan(fim))/Nm)*ds
            fi_A = fi_A + dfip
            lam_A = lam_A + dlamp
            A_ab = A_ab + dAp
        fi_B = fi_A
        lam_B = lam_A
        A_ba = A_ab + math.pi
        
        return fi_B, lam_B, A_ba
    
            
    
    def st_min_s(self,fi_k):
        fi_k = fi_k*180/math.pi # stopnie
        st = math.floor(fi_k)
        min = math.floor((fi_k - st)*60)
        sek = round((fi_k - st - min/60)*3600, 5)
        print(st, 'st', min, 'm', sek, 's')
        
    
    
    
    
    
    
    def vincent(self,a,e2,LAM_A, LAM_B, FI_A, FI_B, A_AB) :
        LAM_B = np.deg2rad(LAM_B)
        FI_B = np.deg2rad(FI_B)
        FI_A = np.deg2rad(FI_A)
        LAM_A = np.deg2rad(LAM_A)
        
        a = 6378137
        e2 = 0.00669438002290
        b = a*(np.sqrt(1-e2))
        f = 1-(b/a)
        dlambda = LAM_B-LAM_A
        UA = math.atan((1-f)*(math.tan(FI_A)))
        UB = math.atan((1-f)*(math.tan(FI_B)))
        L = dlambda
        Ls = 2*L
        while abs(L-Ls)>(0.000001/206265):
            snsigma = np.sqrt((((np.cos(UB))*(np.sin(L)))**2)+(((np.cos(UA))*(np.sin(UB)))-(np.sin(UA))*(np.cos(UB))*(np.cos(L)))**2)
            cssigma = ((np.sin(UA))*(np.sin(UB)))+((np.cos(UA))*(np.cos(UB))*(np.cos(L)))
            sigma = math.atan((snsigma)/(cssigma))
            snalfa = ((np.cos(UA))*(np.cos(UB))*(np.sin(L)))/(snsigma)
            cskwalfa = 1-((snalfa)**2)
            cs2sigmam = cssigma-((2*(np.sin(UA))*(np.sin(UB)))/(cskwalfa))
            C = (f/16)*(cskwalfa)*(4+(f*(4-(3*(cskwalfa)))))
            Ls = L
            L = dlambda+((1-C)*f*(snalfa)*(sigma+(C*(snsigma)*((cs2sigmam)+(C*(cssigma)*(-1+(2*((cs2sigmam)**2))))))))
    
        u2 = (((a**2)-(b**2))/(b**2))*(cskwalfa);
        A = 1+((u2)/16384)*(4096+(u2)*(-768+(u2)*(320-175*(u2))))
        B = ((u2)/1024)*(256+(u2)*(-128+(u2)*(74-(47*(u2)))))
        
        dsigma = B*snsigma*(cs2sigmam+0.25*B*(cssigma*(-1+2*cs2sigmam**2)-(1/6)*B*cs2sigmam*(-3+4*snsigma**2)*(-3+4*cs2sigmam**2)));
        
        S_s = b*A*(sigma-dsigma)
        A_ABs = math.atan(((np.cos(UB))*(np.sin(L)))/(((np.cos(UA))*(np.sin(UB)))-((np.sin(UA))*(np.cos(UB))*(np.cos(L)))))
        A_BAs = math.atan(((np.cos(UA))*(np.sin(L)))/(((-np.sin(UA))*(np.cos(UB)))+((np.cos(UA))*(np.sin(UB))*(np.cos(L)))))+np.pi
    
        if A_ABs < 0:
            A_ABs = A_ABs+2*np.pi
            if A_BAs < 0:
                A_AB = A_AB+2*np.pi 
                if A_BAs> 2*np.pi:
                    A_BAs = A_BAs-2*np.pi
    
        return A_ABs ,A_BAs, S_s
        
    
    def st_m_s (self,A_ABs):
        A_ABs = A_ABs * 180 / np.pi
        st = np.floor(A_ABs)
        m = np.floor((A_ABs - st)*60)
        sek = (A_ABs -st-m/60)*3600
        print(st, 'st', m, 'min', round(sek, 5), 'sek')
        
    
    
    
    
    def uklad_1992(self,fi, lam, a, e2, m_0):
        N = a/(math.sqrt(1-e2 * np.sin(fi)**2))
        t = np.tan(fi)
        n2 = e2 * np.cos(lam)**2
        lam_0 = math.radians(19) #poczatek ukladu w punkcie przeciecia poludnika L0 = 19st z obrazem równika 
        l = lam - lam_0
        
        A_0 = 1 - (e2/4) - (3*(e2**2))/64 - (5*(e2**3))/256
        A_2 = 3/8 * (e2 + ((e2**2)/4) + ((15*e2**3)/128))
        A_4 = 15/256 * (e2**2 + (3*(e2**3))/4)
        A_6 = (35*(e2**3))/3072
        
        sigma = a* ((A_0*fi) - (A_2*np.sin(2*fi)) + (A_4*np.sin(4*fi)) - (A_6*np.sin(6*fi)))
        
        x = sigma + ((l**2)/2) * (N*np.sin(fi)*np.cos(fi)) * (1 + ((l**2)/12) * ((np.cos(fi))**2) * (5 - t**2 + 9*n2 + (4*n2**2)) + ((l**4)/360) * ((np.cos(fi))**4) * (61 - (58*(t**2)) + (t**4) + (270*n2) - (330 * n2 *(t**2))))
        y = l * (N*np.cos(fi)) * (1 + ((((l**2)/6) * (np.cos(fi))**2) * (1-(t**2) + n2)) +  (((l**4)/(120)) * (np.cos(fi)**4)) * (5 - (18 * (t**2)) + (t**4) + (14*n2) - (58*n2*(t**2))))
    
        x92 = round(x * m_0 - 5300000, 3)
        y92 = round(y * m_0 + 500000, 3)   
        
        return x92, y92 
    
    
    
    def uklad_2000(self,fi, lam, a, e2, m_0):
        N = a/(math.sqrt(1-e2 * np.sin(fi)**2))
        t = np.tan(fi)
        n2 = e2 * np.cos(lam)**2
        lam = math.degrees(lam)
        
        if lam > 13.5 and lam < 16.5:
            s = 5
            lam_0 = 15
        elif lam > 16.5 and lam < 19.5:
            s = 6
            lam_0 = 18
        elif lam > 19.5 and lam < 22.5:
            s = 7
            lam_0 = 21
        elif lam > 22.5 and lam < 25.5:
            s = 8
            lam_0 = 24
            
        lam = math.radians(lam)
        lam_0 = math.radians(lam_0)
        l = lam - lam_0
        
        A_0 = 1 - (e2/4) - (3*(e2**2))/64 - (5*(e2**3))/256
        A_2 = 3/8 * (e2 + ((e2**2)/4) + ((15*e2**3)/128))
        A_4 = 15/256 * (e2**2 + (3*(e2**3))/4)
        A_6 = (35*(e2**3))/3072
        
        
        sigma = a* ((A_0*fi) - (A_2*np.sin(2*fi)) + (A_4*np.sin(4*fi)) - (A_6*np.sin(6*fi)))
        
        x = sigma + ((l**2)/2) * (N*np.sin(fi)*np.cos(fi)) * (1 + ((l**2)/12) * ((np.cos(fi))**2) * (5 - t**2 + 9*n2 + (4*n2**2)) + ((l**4)/360) * ((np.cos(fi))**4) * (61 - (58*(t**2)) + (t**4) + (270*n2) - (330 * n2 *(t**2))))
        y = l * (N*np.cos(fi)) * (1 + ((((l**2)/6) * (np.cos(fi))**2) * (1-(t**2) + n2)) +  (((l**4)/(120)) * (np.cos(fi)**4)) * (5 - (18 * (t**2)) + (t**4) + (14*n2) - (58*n2*(t**2))))
    
        x00 = round(x * m_0, 3)
        y00 = round(y * m_0 + (s*1000000) + 500000, 3)   
        
        return x00, y00 




if __name__ == "__main__":
    # utworzenie obiektu
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    X = 3664940.500; Y = 1409153.590; Z = 5009571.170
    phi, lam, h = geo.xyz2plh(X, Y, Z)
    print(phi, lam, h)

        
    
