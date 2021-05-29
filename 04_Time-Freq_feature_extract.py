import numpy as np
from handy import unpickle
from scipy.integrate import simps
from ssqueezepy import cwt
from hurst import compute_Hc
from scipy.stats import skew,kurtosis

#the first 60 lines conserns extracting Holder, the latter consens Hurst and TF_stats
def get_Spectr_manual(run,seg_nr,n_seg=50,co='AG05x'):
    """Pad data, obtain spectogram "manualy", remove padding"""
    def get_wavelet(a, dt=0.01):
        """Get wavelet for wanted frequency"""
        t = np.arange(-3 * a, 3 * a, dt) / a
        return np.exp(1j * 1 * np.pi * t) * np.exp(-t ** 2 / 2)
    pad_len=1000
    neg_pad,pos_pad=pad_len,pad_len
    if seg_nr == 0      : neg_pad, pos_pad = 0,2*pad_len
    if seg_nr == n_seg-1: neg_pad, pos_pad = 2*pad_len,0
    aa=np.linspace(0.1,2,10)
    bb=range(0,1700)
    W=np.zeros((len(aa),len(bb)),dtype=complex)
    x = np.array(unpickle(run=run, n_seg=n_seg,seg_nr=seg_nr, runbyindex=True,pos_pad=pos_pad,neg_pad=neg_pad)[co], dtype='float32')
    for i,a in enumerate(aa):
        wave=get_wavelet(a)
        L=len(wave)
        for j, b in enumerate(bb):
            start=pad_len+b-L//2
            W[i,j]=1/np.sqrt(a)*simps(np.abs(wave*x[start:start+L]))
    return abs(W),aa

def holder(r_i,seg_nr,co):
    """Obtain Holder exponent for every time step"""
    array,aa=get_Spectr_manual(r_i,seg_nr,n_seg=50,co=co)
    a = np.log(aa)
    H=np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        y=np.log(np.abs(array[:,i]))
        H[i]=np.polyfit(a,y,1)[0]
    return H

#----------- INPUT ----------------
n_seg=50
n_runs=7
col=unpickle(column_only=True)
#----------- Initialisation ----------------
mu_holder=np.zeros((n_seg*n_runs,len(col)))
st_holder=np.zeros((n_seg*n_runs,len(col)))
sk_holder=np.zeros((n_seg*n_runs,len(col)))
ku_holder=np.zeros((n_seg*n_runs,len(col)))
#----------- Obtain features----------------
for i,co in enumerate(col):
    for seg_nr_tot in range(0,n_seg*n_runs):
        r_i, seg_nr = seg_nr_tot // n_seg, seg_nr_tot % n_seg
        array=holder(r_i,seg_nr,co)
        mu_holder[seg_nr_tot,i] = np.mean(array)
        st_holder[seg_nr_tot,i] = np.std(array)
        sk_holder[seg_nr_tot,i] = skew(array)
        ku_holder[seg_nr_tot,i] = kurtosis(array)
st_holder/=mu_holder
#
def get_Spectr(run,seg_nr,n_seg=50,co='AG05x'):
    """Pad data, obtain spectogram, remove padding"""
    pad_len=200
    neg_pad,pos_pad=pad_len,pad_len
    if seg_nr == 0      : neg_pad, pos_pad = 0,2*pad_len
    if seg_nr == n_seg-1: neg_pad, pos_pad = 2*pad_len,0
    x = np.array(unpickle(run=run, n_seg=n_seg,seg_nr=seg_nr, runbyindex=True,pos_pad=pos_pad,neg_pad=neg_pad)[co], dtype='float32')
    Spectr= cwt(x, wavelet=('morlet'))[0]
    return abs(Spectr)[:,pad_len:-pad_len]

def get_hurst(array):
    """Obtain Hurst exponent for every time step"""
    Hc=np.zeros(array.shape[1])
    for i in range(array.shape[1]):
        Hc[i],throw,throw2=compute_Hc(array[:,i])#freq,time
    return Hc

#----------- INPUT ----------------
n_seg=50
n_runs=7
col=unpickle(column_only=True)
#----------- Initialisation ----------------

mu_hurst = np.zeros((n_seg * n_runs, len(col)))
st_hurst = np.zeros((n_seg * n_runs, len(col)))
sk_hurst = np.zeros((n_seg * n_runs, len(col)))
ku_hurst = np.zeros((n_seg * n_runs, len(col)))

mu_stat	=	np.zeros((n_seg*n_runs,len(col)))
st_stat	=	np.zeros((n_seg*n_runs,len(col)))
sk_stat	=	np.zeros((n_seg*n_runs,len(col)))
ku_stat	=	np.zeros((n_seg*n_runs,len(col)))
#----------- Obtain features----------------
for i,co in enumerate(col):
    for seg_nr_tot in range(0,n_seg*n_runs):
        r_i, seg_nr = seg_nr_tot // n_seg, seg_nr_tot % n_seg
        array=get_Spectr(r_i,seg_nr,co=co)
        Hc=get_hurst(array)

        mu_hurst[seg_nr_tot,i] = np.mean(Hc)
        st_hurst[seg_nr_tot,i] = np.std(Hc)
        sk_hurst[seg_nr_tot,i] = skew(Hc)
        ku_hurst[seg_nr_tot,i] = kurtosis(Hc)

        mu_stat[seg_nr_tot,i] = np.mean(np.mean(array,axis=1))
        st_stat[seg_nr_tot,i] = np.mean(np.std(array,axis=1))
        sk_stat[seg_nr_tot,i] = np.mean(skew(array,axis=1))
        ku_stat[seg_nr_tot,i] = np.mean(kurtosis(array,axis=1))
st_hurst/=mu_hurst
st_stat/=mu_stat

