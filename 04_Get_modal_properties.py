import numpy as np
from numpy.linalg import svd
import pandas as pd
from scipy.signal import welch,csd
from handy import unpickle

def get_Sxx_mat(run,seg_nr=0,Bins=2,sens_str='g',n_seg=50,runbyindex=False):
    """Creates full power spectral decity matrix"""
    df=unpickle(run=run,n_seg=n_seg,seg_nr=seg_nr,time=False,sensor_group=sens_str,runbyindex=runbyindex)
    df-=df.mean()
    L = df.index[-1] + 1
    for i,sensori in enumerate(df.columns):
        for j,sensorj in enumerate(df.columns):
            if np.logical_and(i==0,j ==0):#intitial run
                f,Sxy=csd(df[sensori],df[sensorj],nperseg=L//Bins,fs=100)
                Sxx_mat=np.zeros([len(df.columns),len(df.columns), len(f[f<30])])
                Sxx_mat[0,0,:]=np.real(Sxy[f<30])
            elif i<=j: # creates lower matrix
                f, Sxy = csd(df[sensori], df[sensorj], nperseg=L//Bins, fs=100)
                Sxx_mat[i,j,:]=np.real(Sxy[f<30])
            else: # creates upper part of matrix by filipping the lower part
                Sxx_mat[i, j, :]=Sxx_mat[j,i,:]
    return f[f<30],Sxx_mat

"""Gives the modeshapes the sime sign as the BC"""
fix_sign = lambda array,eig:array*np.sign(np.dot(array,pd.read_pickle('Shapes_BC')[:,eig]))

def get_SVD_mat(Sxx_mat):
    s_mat=np.zeros([5,Sxx_mat.shape[2]])
    u_mat=np.zeros([Sxx_mat.shape[1],3,Sxx_mat.shape[2]])
    for t in range(Sxx_mat.shape[2]):
        u,s,v=svd(Sxx_mat[:,:,t],compute_uv=True,hermitian=True)
        s_mat[:,t]=s[0:5]
        u_mat[:,:,t]=u[:,:3]
    return s_mat,u_mat

def get_SVD_mode(Sxx_mat,row,poss):
    u,s,v=svd(Sxx_mat[:,:,poss],compute_uv=True,hermitian=True)
    return u[:,row]

def get_MAC(shape,i):
    shape_BC = pd.read_pickle('Shapes_BC')[:, i]
    MAC=np.zeros((3,shape.shape[2]))
    for j in range(3):
        for k in range(shape.shape[2]):
            MAC[j,k]=np.dot(shape[:,j,k],shape_BC)**2/(np.dot(shape[:,j,k],shape[:,j,k])*np.dot(shape_BC,shape_BC))
    return MAC

def normalize_store_BC_all():
    shape_BC = pd.read_pickle('Shapes_BC')  # mode,eig_i
    for i in range(shape_BC.shape[1]):
        shape_BC[0:40, i] /= np.linalg.norm(shape_BC[0:40, i])
        shape_BC[40:58, i] /= np.linalg.norm(shape_BC[40:58, i])
        shape_BC[58:, i] /= np.linalg.norm(shape_BC[58:, i])
    pd.to_pickle(shape_BC, 'Shapesn_BC')
    shape_all = pd.read_pickle('all_mode_data')  # seg,eig_i,mode
    for seg_nr_tot in range(shape_all.shape[0]):
        for i in range(shape_all.shape[1]):
            shape_all[seg_nr_tot, i, 0:40] /= np.linalg.norm(shape_all[seg_nr_tot, i, 0:40])
            shape_all[seg_nr_tot, i, 40:58] /= np.linalg.norm(shape_all[seg_nr_tot, i, 40:58])
            shape_all[seg_nr_tot, i, 58:] /= np.linalg.norm(shape_all[seg_nr_tot, i, 58:])
    pd.to_pickle(shape_all, 'Shapesn_all')
    return
def createBC():
    # ----------- INPUT ----------------
    runs, Bins, n_seg, eigs_BC, eigs_row,eigs_row_peak, eigs_delta = init(createBC=True)
    r_i, run=0,2
    # ----------- Obtaining FDD info ----------------
    f, Sxx_mat = get_Sxx_mat(run, 0, sens_str='all', n_seg=1,Bins=Bins)
    s_mat, u_mat = get_SVD_mat(Sxx_mat)
    eigs = np.zeros(len(eigs_BC))
    Shapes = np.zeros((u_mat.shape[0], len(eigs)))
    # ----------- Peak picking ----------------
    for i, eig in enumerate(eigs_BC):
        logic = np.logical_and(f > eig - eigs_delta[i], f < eig + eigs_delta[i])
        s_temp = s_mat[eigs_row_peak[i], logic]
        pos = np.argmax(s_temp)
        fpos = np.where(logic)[0][0] + pos
        eigs[i]=f[fpos]
        Shapes[:, i]=u_mat[:,eigs_row[i],fpos]

    #improved56=pd.read_pickle('Shapes_g_BC')  Modeshape 5 and 6 had no forced orthogonality when only global sensors where used
    #Shapes[-35:,5]=improved56[:,5] #These was 1 prinsipal, with no forced orthogonality
    #Shapes[-35:,6] = improved56[:,6]#obtained by small tweeks to the program
    #Shapes[:,5]/=np.sum(Shapes[:,5]**2)
    #Shapes[:,6]/=np.sum(Shapes[:,6]**2)
    pd.to_pickle(Shapes, 'Shapes_BC')
    pd.to_pickle(eigs, 'eigf_BC')
    return

def init(createBC=False):
    eigs_BC = [3.11764706, 6.05882353, 6.88235294, 7.23529412, 7.35294118,8.05882353,
               8.64705882, 9.41176471, 10.70588235, 12.82352941, 18.29411765, 24.11764706]
    eigs_row_peak = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
    eigs_row =      [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    eigs_delta=[0.4,0.4,0.3,0.4,0.4,0.4,0.4,0.4,0.4,2,2,4]
    eigs=np.copy(eigs_BC)
    runs=[2,5,11,23,26,29,32]
    Bins=2
    n_seg=50
    if createBC:runs,Bins,n_seg=[2],50,1
    return runs,Bins,n_seg,eigs,eigs_row,eigs_row_peak,eigs_delta

createBC() #this function creates BC modeshapes
#----------- INPUT ----------------
runs,Bins,n_seg,eigs_BC,eigs_row,eigs_row_peak,eigs_delta=init(createBC=False)
#----------- Initialisation ----------------
store_mode=np.zeros((len(runs)*n_seg,len(eigs_BC),len(unpickle(sensor_group='all',column_only=True))))
store_s=np.zeros((len(runs)*n_seg,len(eigs_BC)))
store_eigf=np.zeros((len(runs)*n_seg,len(eigs_BC)))
for seg_nr_tot in range(len(runs)*n_seg):
    # ----------- Obtaining FDD info ----------------
    r_i, seg_nr = seg_nr_tot // n_seg, seg_nr_tot % n_seg
    f,Sxx_mat=get_Sxx_mat(r_i,seg_nr=seg_nr,Bins=Bins,sens_str='all',n_seg=n_seg,runbyindex=True)
    s_mat,u_mat=get_SVD_mat(Sxx_mat) #Obtain all SVD data from all frequencies

    # ----------- Peak picking ----------------
    for i,eig in enumerate(eigs_BC):
        logic=np.logical_and(f>eig-eigs_delta[i],f<eig+eigs_delta[i]) #include only frequencies in vicinity of peak
        MAC=get_MAC(u_mat[:,:,logic],i)
        row,pos=np.argmax(MAC)//sum(logic),np.argmax(MAC)%sum(logic) #Finds what MAC number is highest
        store_eigf[seg_nr_tot,i]=f[np.where(logic)[0][0]+pos]
        store_s   [seg_nr_tot,i]=s_mat[row,np.where(logic)[0][0] + pos]
        store_mode[seg_nr_tot,i,:]=u_mat[:,row,np.where(logic)[0][0]+pos]
        store_mode[seg_nr_tot,i,:]=fix_sign(store_mode[seg_nr_tot,i,:],i)


pd.to_pickle(store_mode,'Shapes_all')
pd.to_pickle(store_s,'s_all')
pd.to_pickle(store_eigf,'eigf_all')

normalize_store_BC_all()


