import numpy as np
from numpy.linalg import svd
import pandas as pd
from scipy.signal import welch
from math import sqrt
from handy import unpickle
from scipy import integrate


#----------- Modal_features ----------------
#These functions get mode features from the files:
#Shapesn/Shapes_all/BC,eigf_all/BC,s_all
#Where all denotes all cases and BC denotes base cases.
#Shapesn are normalised after sensor groups, shapes are nomalised after all sensors

def get_direct():
    def do_stuff(shape_all):
        dir = np.zeros((shape_all.shape[0], (shape_all.shape[1]*shape_all.shape[2])))
        L=shape_all.shape[2]
        for seg_nr_tot in range(shape_all.shape[0]):
            for i in range(shape_all.shape[1]):
                dir[seg_nr_tot,i*L:(i+1)*L]=shape_all[seg_nr_tot,i,:]
        return dir
    df= pd.dataframe(do_stuff(pd.read_pickle('Shapes_all')))
    #dirn= pd.dataframe(do_stuff(pd.read_pickle('Shapesn_all')))
    return df, #dirn

def get_MAC():
    get_MAC_2 = lambda array1, array2: np.dot(array1, array2) ** 2 / (np.dot(array1, array1) * np.dot(array2, array2))
    shape_all = pd.read_pickle('Shapesn_all')
    shape_BC = pd.read_pickle('Shapesn_BC')
    MAC_l = np.zeros((shape_all.shape[0], shape_all.shape[1]))
    MAC_gx = np.zeros((shape_all.shape[0], shape_all.shape[1]))
    MAC_gz = np.zeros((shape_all.shape[0], shape_all.shape[1]))
    for seg_nr_tot in range(shape_all.shape[0]):
        for eig_i in range(shape_all.shape[1]):
            MAC_l[seg_nr_tot,eig_i]=get_MAC_2(shape_all[seg_nr_tot, eig_i, 0:40],shape_BC[0:40,eig_i])
            MAC_gx[seg_nr_tot,eig_i]=get_MAC_2(shape_all[seg_nr_tot, eig_i, 40:58],shape_BC[40:58,eig_i])
            MAC_gz[seg_nr_tot,eig_i]=get_MAC_2(shape_all[seg_nr_tot, eig_i, 58:],shape_BC[58:,eig_i])
    df_l = pd.DataFrame(data=MAC_l)
    df_gx = pd.DataFrame(data=MAC_gx)
    df_gz = pd.DataFrame(data=MAC_gz)
    return df_l,df_gx,df_gz
def get_eigf():
    data = pd.read_pickle('eigf_all') 
    col=['Eig_Freq_1','Eig_Freq_2','Eig_Freq_3','Eig_Freq_4','Eig_Freq_5','Eig_Freq_6','Eig_Freq_7','Eig_Freq_8','Eig_Freq_9','Eig_Freq_10','Eig_Freq_11','Eig_Freq_12']
    df=pd.DataFrame(data=data,columns=col)
    return df
def get_COMAC():
    def get_MAC_2(array1, array2):
        return np.dot(array1, array2) ** 2 / (np.dot(array1, array1) * np.dot(array2, array2))
    shape_all = pd.read_pickle('Shapes_all')
    shape_BC = pd.read_pickle('Shapes_BC')
    COMAC = np.zeros((shape_all.shape[0], shape_all.shape[2]))
    for seg_nr_tot in range(shape_all.shape[0]):
        for sensor in range(shape_all.shape[2]):
            COMAC[seg_nr_tot,sensor]=get_MAC_2(shape_all[seg_nr_tot, :, sensor],shape_BC[sensor,:])
    df=pd.DataFrame(data=COMAC)
    pd.to_pickle(df,'feature/COMAC')
    return df
def get_yuen():
    eig_all= pd.read_pickle('eigf_all')
    shape_all = pd.read_pickle('Shapes_all')
    shape_BC = pd.read_pickle('Shapes_BC')
    eig_BC=pd.read_pickle('eigf_BC')
    yuen=np.zeros((shape_all.shape[0],shape_all.shape[1]*shape_all.shape[2]))
    for seg_nr_tot in range(shape_all.shape[0]):
            for eig_i in range(shape_all.shape[1]):
                yuen[seg_nr_tot,shape_all.shape[2]*eig_i:shape_all.shape[2]*(eig_i+1)]=np.abs(shape_all[seg_nr_tot,eig_i,:]/eig_all[seg_nr_tot,eig_i])-np.abs(shape_BC[:,eig_i]/eig_BC[eig_i])
    df=pd.DataFrame(data=yuen)
    return df
def get_curve_stainU_flexG():
    shape_all = pd.read_pickle('Shapesn_all')[:,:,0:40]
    shape_all_unn = pd.read_pickle('Shapes_all')[:, :, 0:40]
    s_mat=pd.read_pickle('s_all')
    eigf_BC = pd.read_pickle('eigf_BC')
    curve=np.zeros((shape_all.shape[0],3*10))
    U=np.zeros((shape_all.shape[0],3))
    G=np.zeros((shape_all.shape[0], 10))
    for seg_nr_tot in range(shape_all.shape[0]):
        for j,eig_i in enumerate([2,10,11]):
            i=0
            fi_1 = np.mean((shape_all[seg_nr_tot, eig_i, i * 4:(i + 1) * 4]))
            fi_2 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 1) * 4:(i + 2) * 4]))
            fi_3 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 2) * 4:(i + 3) * 4]))
            curve[seg_nr_tot, j * 10 + i] = (3*fi_1 - 4 * fi_2+ fi_3 )/2
            i=7
            fi_1 = np.mean((shape_all[seg_nr_tot, eig_i, i * 4:(i + 1) * 4]))
            fi_2 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 1) * 4:(i + 2) * 4]))
            fi_3 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 2) * 4:(i + 3) * 4]))
            curve[seg_nr_tot, j * 10 + i+2] = (1*fi_1 - 4 * fi_2+ 3*fi_3 )/2
            for i in range(0,8):
                fi_1=np.mean((shape_all[seg_nr_tot,eig_i,i*4:(i+1)*4]))
                fi_2=np.mean((shape_all[seg_nr_tot,eig_i,(i+1)*4:(i+2)*4]))
                fi_3=np.mean((shape_all[seg_nr_tot,eig_i,(i+2)*4:(i+3)*4]))
                curve[seg_nr_tot,j*10+i+1]=fi_1-2*fi_2+fi_3
            for i in range(0, 10):
                array=shape_all[seg_nr_tot, eig_i, i * 4:(i + 1) * 4]
                G[seg_nr_tot,i]+=((np.sum(array)/4)**2)/(eigf_BC[eig_i]**2)
            U[seg_nr_tot, j]=np.matmul(np.transpose(shape_all_unn[seg_nr_tot,eig_i,0:40]),shape_all_unn[seg_nr_tot,eig_i,0:40])*s_mat[seg_nr_tot,eig_i]**2
    df_curve=pd.DataFrame(data=curve)
    df_U=pd.DataFrame(data=U)
    df_G=pd.DataFrame(data=G)
    return df_curve,df_U,df_G
#----------- NonModal_features ----------------
def get_low_med_hig_std():
    lim1 = 5
    lim2 = 20

    low_freq = np.zeros([350, len(unpickle(column_only=True))])
    med_freq = np.zeros([350, len(unpickle(column_only=True))])
    high_freq = np.zeros([350, len(unpickle(column_only=True))])
    for seg_nr_tot in range(350):
        print(seg_nr_tot // 50, seg_nr_tot % 50)
        df = unpickle(run=seg_nr_tot // 50, n_seg=50, seg_nr=seg_nr_tot % 50, time=False, runbyindex=True)
        df = df - df.mean()
        L = df.index[-1]
        for j, sensor in enumerate(df.columns):
            f, Sxx_f = welch(df[sensor], nperseg=L // 2, fs=100)

            dx = f[1] - f[0]
            Sxx_low = Sxx_f[f < lim1]
            Sxx_med = Sxx_f[np.logical_and(f > lim1, f < lim2)]
            Sxx_high = Sxx_f[f > lim2]
            a = sqrt(integrate.trapz(Sxx_low, dx=dx))
            b = sqrt(integrate.trapz(Sxx_med, dx=dx))
            c = sqrt(integrate.trapz(Sxx_high, dx=dx))
            low_freq[seg_nr_tot, j] = a / (a + b + c)  # sum(std)=1 might not be recomended
            med_freq[seg_nr_tot, j] = b / (a + b + c)  # compared to sum(var)=1
            high_freq[seg_nr_tot, j] = c / (a + b + c)
    return pd.DataFrame(data=Sxx_low),pd.DataFrame(data=Sxx_med),pd.DataFrame(data=Sxx_high)

def get_pca_res():
    #Note that this function imports All_Sxx, it is the auto spectrums.
    # These were obtain in a similar manner Sxx_f in the "get_low_med_hig_std" function

    def normalize(array): #normalizes each Sxixi to length 1
        def normalize_2(array, l, col):
            for seg_nr_tot in range(350):
                for i, co in enumerate(col):
                    array[seg_nr_tot, i * l:(i + 1) * l] /= np.linalg.norm(array[seg_nr_tot, i * l:(i + 1) * l])
            return array
        return normalize_2(array, array.shape[1] // 75, np.array(unpickle(column_only=True)))
    def lf_svd(array):  #Use svd to get base case Sxx shapes
        l = array.shape[1] // 75
        col = unpickle(column_only=True)
        u_mat = np.zeros((array.shape[1], 50))
        for i, co in enumerate(col):
            u, s, v = svd(np.transpose(array[0:50, i * l:(i + 1) * l]), compute_uv=True)
            u_mat[i * l:(i + 1) * l, :] = u[:, :50]
        pd.to_pickle(u_mat, 'LF_spectra')
        return u_mat

    def feat_svd(): #get the values
        col = np.array(unpickle(column_only=True))
        u_mat = lf_svd(normalize(pd.read_pickle('All_Sxx'))) #
        Sxx = normalize(pd.read_pickle('All_Sxx'))
        l = Sxx.shape[1] // 75
        r = np.zeros((350, len(col), 50))
        for seg_nr_tot in range(350):
            for i, co in enumerate(col):
                for r_i in range(50):
                    r[seg_nr_tot, i, r_i] = -np.dot(u_mat[i * l:(i + 1) * l, r_i], Sxx[seg_nr_tot, i * l:(i + 1) * l])
        return np.square(r)
    r = feat_svd()
    feat1=1-np.sum(r[:,:,:1],axis=-1)
    feat18=1-np.sum(r[:, :, :18],axis=-1)
    return pd.DataFrame(data=feat1),pd.DataFrame(data=feat18)
