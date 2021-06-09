import numpy as np
from handy import unpickle
from scipy.integrate import simps
from ssqueezepy import cwt
from hurst import compute_Hc
from scipy.stats import skew,kurtosis
import pandas as pd

#the first 70 lines conserns extracting Holder, the latter consens Hurst and TF_stats
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
def create_holder_files():
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
    df = pd.DataFrame(data=mu_holder, columns=col)
    pd.to_pickle(df, 'feature/holder_mu')
    df = pd.DataFrame(data=st_holder/mu_holder, columns=col)
    pd.to_pickle(df, 'feature/holder_st')
    df = pd.DataFrame(data=sk_holder, columns=col)
    pd.to_pickle(df, 'feature/holder_sk')
    df = pd.DataFrame(data=ku_holder, columns=col)
    pd.to_pickle(df, 'feature/holder_ku')




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
def create_hurst_stat_files():
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
    pd.to_pickle(mu_hurst, 'feature/hurst_mu')
    pd.to_pickle(st_hurst/mu_hurst, 'feature/hurst_st')
    pd.to_pickle(sk_hurst, 'feature/hurst_sk')
    pd.to_pickle(ku_hurst, 'feature/hurst_ku')
    pd.to_pickle(mu_stat, 'feature/stat_mu')
    pd.to_pickle(st_stat/mu_stat, 'feature/stat_st')
    pd.to_pickle(sk_stat, 'feature/stat_sk')
    pd.to_pickle(ku_stat, 'feature/stat_ku')

#----------- Create features ----------------
#These are the feature extraction, these take several days
create_hurst_stat_files()
create_holder_files()
#----------- Combine features in file ----------------
element=unpickle(sensor_group='all',column_only='True')
df1 = pd.read_pickle('feature/time_freq_moment_st_mu')
df2 = pd.read_pickle('feature/time_freq_moment_sk')
df3 = pd.read_pickle('feature/time_freq_moment_ku')
df4 = pd.read_pickle('feature/holder_mu')
df5 = pd.read_pickle('feature/holder_st_mu')
df6 = pd.read_pickle('feature/holder_sk')
df7 = pd.DataFrame(pd.read_pickle('feature/holder_ku'))
df8 = pd.DataFrame(pd.read_pickle('feature/hurst_mu'))
df9 = pd.DataFrame(pd.read_pickle('feature/hurst_st_mu'))
df10 = pd.DataFrame(pd.read_pickle('feature/hurst_sk'))
df11 = pd.DataFrame(pd.read_pickle('feature/hurst_ku'))
col=[None]*(11)
col[:3]=['TF_' + str(s) for s in ['STD','Skewness','Kurtosis']]
col[3:7]=['Holder_' + str(s) for s in ['Mean','STD','Skewness','Kurtosis']]
col[7:] = ['Hurst_' + str(s) for s in ['Mean','STD','Skewness','Kurtosis']]
for i,el in enumerate(element):
    df = pd.concat([df1[i], df2[i], df3[i], df4[el],df5[el],df6[el],df7[el],df8[i],df9[i],df10[i],df11[i]],axis=1, join='outer')#
    df.columns=col
    pd.to_pickle(df,'Time_Freq_output_' + str(50) +'_'+ el + '.pkl')
