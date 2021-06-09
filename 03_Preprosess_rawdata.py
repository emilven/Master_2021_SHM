from nptdms import TdmsFile
import numpy as np
import pandas as pd
from scipy.signal import decimate,detrend

# In line 108, input file path of measurement data

def readtdms(path):
    """From TDMS file to pandas dataframe"""
    with TdmsFile.read(path) as tdms_file:
        group = tdms_file['acceleration_data']
        col = [i[22:-1] for i in list(tdms_file._channel_data.keys())]
        return pd.DataFrame(data=np.transpose(np.asmatrix([np.array(group[j][:]) for i, j in enumerate(col)])), columns=col)
def downsample(df):
    """Downsample from 400 to 100 Hz"""
    col=df.columns
    [a,b]=df.shape
    matrix=np.zeros([int(round(a/4)),b])
    for i in range(b):
        matrix[:,i]=decimate(df.iloc[:,i],4, axis=0)
        df.drop(df.columns[0], axis=1)
    return fixtime(pd.DataFrame(data=matrix, columns=col),100)
def fixtime(df,f=100):
    """makes time go from 0, with step of 1/f"""
    df.timestamp = np.linspace(0, (df.index[-1]) /f,num=df.index[-1]+1, endpoint=True)
    return df
def volt_to_acc(df):
    """from mV to m/s^2"""
    df.AL01 *= 9.81 / 491.75
    df.AL02 *= 9.81 / 490.07
    df.AL03 *= 9.81 / 479.66
    df.AL04 *= 9.81 / 482.1
    df.AL05 *= 9.81 / 515.1
    df.AL06 *= 9.81 / 486.65
    df.AL07 *= 9.81 / 506.29
    df.AL08 *= 9.81 / 487.99
    df.AL09 *= 9.81 / 498.13
    df.AL10 *= 9.81 / 516.89
    df.AL11 *= 9.81 / 503.95
    df.AL12 *= 9.81 / 518.57
    df.AL13 *= 9.81 / 507.37
    df.AL14 *= 9.81 / 494.42
    df.AL15 *= 9.81 / 494.45
    df.AL16 *= 9.81 / 490.56
    df.AL17 *= 9.81 / 497.11
    df.AL18 *= 9.81 / 498.88
    df.AL19 *= 9.81 / 506.47
    df.AL20 *= 9.81 / 499.7
    df.AL21 *= 9.81 / 495.67
    df.AL22 *= 9.81 / 495.02
    df.AL23 *= 9.81 / 497.87
    df.AL24 *= 9.81 / 490.8
    df.AL25 *= 9.81 / 497.97
    df.AL26 *= 9.81 / 490.8
    df.AL27 *= 9.81 / 496.64
    df.AL28 *= 9.81 / 490.05
    df.AL29 *= 9.81 / 506.64
    df.AL30 *= 9.81 / 493.94
    df.AL31 *= 9.81 / 486.15
    df.AL32 *= 9.81 / 493.91
    df.AL33 *= 9.81 / 493.98
    df.AL34 *= 9.81 / 487.03
    df.AL35 *= 9.81 / 485.1
    df.AL36 *= 9.81 / 491.6
    df.AL37 *= 9.81 / 492.51
    df.AL38 *= 9.81 / 493.67
    df.AL39 *= 9.81 / 492.36
    df.AL40 *= 9.81 / 492.19
    df.AG01x *= 9.81 / 520.78
    df.AG01z *= 9.81 / 479.26
    df.AG02x *= 9.81 / 1003.53
    df.AG02z *= 9.81 / 1041.94
    df.AG03x *= 9.81 / 518.99
    df.AG03z *= 9.81 / 457.8
    df.AG04x *= 9.81 / 1052.36
    df.AG04z *= 9.81 / 1033.53
    df.AG05x *= 9.81 / 514.64
    df.AG05z *= 9.81 / 493.66
    df.AG06x *= 9.81 / 994.26
    df.AG06z *= 9.81 / 1010.84
    df.AG07x *= 9.81 / 499.19
    df.AG07z *= 9.81 / 492.82
    df.AG08x *= 9.81 / 513.67
    df.AG08z *= 9.81 / 474.59
    df.AG09x *= 9.81 / 527.81
    df.AG10x *= 9.81 / 501.66
    df.AG10z *= 9.81 / 470.51
    df.AG11x *= 9.81 / 486.93
    df.AG11z *= 9.81 / 496.62
    df.AG12x *= 9.81 / 1014.75
    df.AG12z *= 9.81 / 1036.02
    df.AG13x *= 9.81 / 489.28
    df.AG13z *= 9.81 / 472.13
    df.AG14x *= 9.81 / 1025.86
    df.AG14z *= 9.81 / 1048.98
    df.AG15x *= 9.81 / 492.32
    df.AG15z *= 9.81 / 497.25
    df.AG16x *= 9.81 / 1029.69
    df.AG16z *= 9.81 / 1045.82
    df.AG17x *= 9.81 / 512.83
    df.AG17z *= 9.81 / 484.69
    df.AG18x *= 9.81 / 488.55
    df.AG18z *= 9.81 / 476.5
    df.AS *= 9.81 / 99.1
    return df

numberlist=['02','05','11','23','26','29','32']
loc=r"C:\Users\mats-\Documents\Masteroppgave 2\Masteroppgave 2\_03_Data"
for i in numberlist:
    df = downsample(volt_to_acc(readtdms(loc + "\MVS_P2_RUN" + str(i) + "_ACCELERATION.tdms")))[10250:92750] #Obtain 100Hz data
    for c in df.columns[1:]: df[c] = detrend(df[c]) #Linear detrening
    df = fixtime(df.reset_index(drop=True)) #Index and time fixing
    df.to_pickle("RUN"+str(i)+"_100Hz_cut") #final timeseries format

