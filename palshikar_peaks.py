__author__ = 'hok1'

# G. K. Palshikar, "Simple Algorithms for Peak Detection in Time-Series"
# D. T. Pele, "An LPPL algorithm for estimating the critical time of a stock market bubble"

import numpy as np

def S1(k, i, parray):
    if i < 0  or i >= len(parray):
        raise IndexError()
    currp = parray[i]
    leftarray = currp-parray[max(0, i-k):i]
    rightarray = currp-parray[(i+1):min(i+k+1,len(parray))]
    return 0.5*((np.max(leftarray) if len(leftarray)>0 else 0)+(np.max(rightarray) if len(rightarray)>0 else 0))

def peaks(parray, k, h=1.5):
    a = np.array(map(lambda i: S1(k, i, parray), range(len(parray))))
    m = np.mean(a)
    s = np.std(a)
    peakidx = []
    for idx in range(len(a)):
        if a[idx]>0 and (a[idx]-m)>h*s:
            peakidx.append(idx)
    peakidx = sorted(peakidx)
    jdx = 0
    while jdx<len(peakidx)-1:
        if peakidx[jdx+1]-peakidx[jdx] < k:
            if parray[peakidx[jdx]]<parray[peakidx[jdx+1]]:
                del peakidx[jdx]
            else:
                del peakidx[jdx+1]
        else:
            jdx += 1
    return peakidx