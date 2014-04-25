import palshikar_peaks as peak
import numpy as np

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
        for m in range(len(yarray)):
            peaks = peak.peaks(yarray, 3, h=1.5)
            if len(peaks) < 3:
                continue
            i = np.random.randint(0, len(peaks)-2)
            j = i+1
            k = i+2

            rho = float(j-i)/float(k-j)
            tc = (rho*k-j)/(rho-1)
            omega = 2*np.pi/np.log(rho)
            phi = np.pi - omega*np.log(tc-k)