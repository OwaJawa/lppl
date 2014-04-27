import palshikar_peaks as peak
import numpy as np
import lpplmodel

class LPPLPeleAlgorithm:
    def __init__(self):
        self.meanA = 600
        self.meanB = -250
        self.meanC = -20
        self.stdA = 50
        self.stdB = 50
        self.stdC = 25
        self.stdtc = 5

    def rolling_peaks_price_gyrations(self, tarray, yarray):
        delta_t = np.mean([tarray[i+1]-tarray[i] for i in range(len(tarray)-1)])
        z = 0.5
        for m in range(len(yarray)):
            peaks = peak.peaks(yarray, 3, h=1.5)
            if len(peaks) < 3:
                continue
            i = np.random.randint(0, len(peaks)-2)
            j = i+1
            k = i+2

            rho = float(tarray[j]-tarray[i])/float(tarray[k]-tarray[j])
            tc = (rho*tarray[k]-tarray[j])/(rho-delta_t)
            omega = 2*np.pi/np.log(rho)
            phi = np.pi - omega*np.log(tc-tarray[k])

            ols_sol = self.solve_linear_parameters(tarray, yarray, tc, z, omega, phi)
            A = ols_sol['A']
            B = ols_sol['B']
            C = ols_sol['C']

            # applying LMA here

            sol = {'A': A, 'B': B, 'C': C, 'z': z, 'omega': omega, 'tc': tc, 'phi': phi, }

    def solve_linear_parameters(self, tarray, yarray, tc, z, omega, Phi):
        f = lambda t: (tc-t)**z if tc>=t else (t-tc)**z
        g = lambda t: np.cos(omega*np.log(tc-t if tc>=t else t-tc)+Phi)
        A = np.matrix(np.zeros([3, 3]))
        b = np.matrix(np.zeros([3, 1]))

        farray = np.array(map(f, tarray))
        garray = np.array(map(g, tarray))
        A[0, 0] = len(tarray)
        A[0, 1] = np.sum(farray)
        A[0, 2] = np.sum(garray)
        A[1, 0] = A[0, 1]
        A[1, 1] = np.sum(farray*farray)
        A[1, 2] = np.sum(farray*garray)
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        A[2, 2] = np.sum(garray*garray)

        b[0, 0] = np.sum(yarray)
        b[1, 0] = np.sum(yarray*farray)
        b[2, 0] = np.sum(yarray*garray)

        sol = np.linalg.pinv(A)*b
        return {'A': sol[0, 0], 'B': sol[1, 0], 'C': sol[2, 0]}