import numpy as np
from numpy.linalg import svd
import pandas as pd
from scipy.signal import welch
from math import sqrt
from handy import unpickle
from scipy import integrate


# ----------- Modal_features ----------------
# These functions get mode features from the files:
# Shapesn/Shapes_all/BC,eigf_all/BC,s_all
# Where all denotes all cases and BC denotes base cases.
# These are made by running 04_Get_modal_properties.py
# Shapesn are normalised after sensor groups, shapes are nomalised after all sensors
# After line 215 they are extracted and collected

# ----------- Functions that calculate features ----------------
def get_direct():  # Obdtains direct modeshapes
    def do_stuff(shape_all):
        dir = np.zeros((shape_all.shape[0], (shape_all.shape[1] * shape_all.shape[2])))
        L = shape_all.shape[2]
        for seg_nr_tot in range(shape_all.shape[0]):
            for i in range(shape_all.shape[1]):
                dir[seg_nr_tot, i * L:(i + 1) * L] = shape_all[seg_nr_tot, i, :]
        return dir

    df = pd.DataFrame((do_stuff(pd.read_pickle('Shapes_all'))))
    # dirn= pd.DataFrame(do_stuff(pd.read_pickle('Shapesn_all')))
    return df  # ,dirn


def get_MAC():  # Obdtains MAC
    get_MAC_2 = lambda array1, array2: np.dot(array1, array2) ** 2 / (np.dot(array1, array1) * np.dot(array2, array2))
    shape_all = pd.read_pickle('Shapesn_all')
    shape_BC = pd.read_pickle('Shapesn_BC')
    MAC_l = np.zeros((shape_all.shape[0], shape_all.shape[1]))
    MAC_gx = np.zeros((shape_all.shape[0], shape_all.shape[1]))
    MAC_gz = np.zeros((shape_all.shape[0], shape_all.shape[1]))
    for seg_nr_tot in range(shape_all.shape[0]):
        for eig_i in range(shape_all.shape[1]):
            MAC_l[seg_nr_tot, eig_i] = get_MAC_2(shape_all[seg_nr_tot, eig_i, 0:40], shape_BC[0:40, eig_i])
            MAC_gx[seg_nr_tot, eig_i] = get_MAC_2(shape_all[seg_nr_tot, eig_i, 40:58], shape_BC[40:58, eig_i])
            MAC_gz[seg_nr_tot, eig_i] = get_MAC_2(shape_all[seg_nr_tot, eig_i, 58:], shape_BC[58:, eig_i])
    list = ['MAC_1', 'MAC_2', 'MAC_3', 'MAC_4', 'MAC_5', 'MAC_6', 'MAC_7', 'MAC_8', 'MAC_9', 'MAC_10', 'MAC_11',
            'MAC_12']
    a = [s[:3] + '_lx' + s[3:] for s in list]
    b = [s[:3] + '_gx' + s[3:] for s in list]
    c = [s[:3] + '_gz' + s[3:] for s in list]
    df_l = pd.DataFrame(data=MAC_l, columns=a)
    df_gx = pd.DataFrame(data=MAC_gx, columns=b)
    df_gz = pd.DataFrame(data=MAC_gz, columns=c)
    return df_l, df_gx, df_gz


def get_eigf():  # Obdtains eigenfrequencies
    data = pd.read_pickle('eigf_all')
    col = ['Eig_Freq_1', 'Eig_Freq_2', 'Eig_Freq_3', 'Eig_Freq_4', 'Eig_Freq_5', 'Eig_Freq_6', 'Eig_Freq_7',
           'Eig_Freq_8', 'Eig_Freq_9', 'Eig_Freq_10', 'Eig_Freq_11', 'Eig_Freq_12']
    df = pd.DataFrame(data=data, columns=col)
    return df


def get_COMAC():  # Obdtains COMAC
    def get_MAC_2(array1, array2):
        return np.dot(array1, array2) ** 2 / (np.dot(array1, array1) * np.dot(array2, array2))

    shape_all = pd.read_pickle('Shapes_all')
    shape_BC = pd.read_pickle('Shapes_BC')
    COMAC = np.zeros((shape_all.shape[0], shape_all.shape[2]))
    for seg_nr_tot in range(shape_all.shape[0]):
        for sensor in range(shape_all.shape[2]):
            COMAC[seg_nr_tot, sensor] = get_MAC_2(shape_all[seg_nr_tot, :, sensor], shape_BC[sensor, :])
    df = pd.DataFrame(data=COMAC)
    # pd.to_pickle(df, 'feature/COMAC')
    return df


def get_yuen():  # Obdtains yuen function
    eig_all = pd.read_pickle('eigf_all')
    shape_all = pd.read_pickle('Shapes_all')
    shape_BC = pd.read_pickle('Shapes_BC')
    eig_BC = pd.read_pickle('eigf_BC')
    yuen = np.zeros((shape_all.shape[0], shape_all.shape[1] * shape_all.shape[2]))
    for seg_nr_tot in range(shape_all.shape[0]):
        for eig_i in range(shape_all.shape[1]):
            yuen[seg_nr_tot, shape_all.shape[2] * eig_i:shape_all.shape[2] * (eig_i + 1)] = np.abs(
                shape_all[seg_nr_tot, eig_i, :] / eig_all[seg_nr_tot, eig_i]) - np.abs(
                shape_BC[:, eig_i] / eig_BC[eig_i])
    df = pd.DataFrame(data=yuen)
    return df


def get_curve_stainU_flexG():  # Obdtains Curvature, Strain energy, and Flexability
    shape_all = pd.read_pickle('Shapesn_all')[:, :, 0:40]
    shape_all_unn = pd.read_pickle('Shapes_all')[:, :, 0:40]
    s_mat = pd.read_pickle('s_all')
    eigf_BC = pd.read_pickle('eigf_BC')
    curve = np.zeros((shape_all.shape[0], 3 * 10))
    U = np.zeros((shape_all.shape[0], 3))
    G = np.zeros((shape_all.shape[0], 10))
    for seg_nr_tot in range(shape_all.shape[0]):
        for j, eig_i in enumerate([2, 10, 11]):
            i = 0
            fi_1 = np.mean((shape_all[seg_nr_tot, eig_i, i * 4:(i + 1) * 4]))
            fi_2 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 1) * 4:(i + 2) * 4]))
            fi_3 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 2) * 4:(i + 3) * 4]))
            curve[seg_nr_tot, j * 10 + i] = (3 * fi_1 - 4 * fi_2 + fi_3) / 2
            i = 7
            fi_1 = np.mean((shape_all[seg_nr_tot, eig_i, i * 4:(i + 1) * 4]))
            fi_2 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 1) * 4:(i + 2) * 4]))
            fi_3 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 2) * 4:(i + 3) * 4]))
            curve[seg_nr_tot, j * 10 + i + 2] = (1 * fi_1 - 4 * fi_2 + 3 * fi_3) / 2
            for i in range(0, 8):
                fi_1 = np.mean((shape_all[seg_nr_tot, eig_i, i * 4:(i + 1) * 4]))
                fi_2 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 1) * 4:(i + 2) * 4]))
                fi_3 = np.mean((shape_all[seg_nr_tot, eig_i, (i + 2) * 4:(i + 3) * 4]))
                curve[seg_nr_tot, j * 10 + i + 1] = fi_1 - 2 * fi_2 + fi_3
            for i in range(0, 10):
                array = shape_all[seg_nr_tot, eig_i, i * 4:(i + 1) * 4]
                G[seg_nr_tot, i] += ((np.sum(array) / 4) ** 2) / (eigf_BC[eig_i] ** 2)
            U[seg_nr_tot, j] = np.matmul(np.transpose(shape_all_unn[seg_nr_tot, eig_i, 0:40]),
                                         shape_all_unn[seg_nr_tot, eig_i, 0:40]) * s_mat[seg_nr_tot, eig_i] ** 2
    col = ['Curve_v1_0', 'Curve_v1_1', 'Curve_v1_2', 'Curve_v1_3', 'Curve_v1_4', 'Curve_v1_5', 'Curve_v1_6',
           'Curve_v1_7', 'Curve_v1_8', 'Curve_v1_9',
           'Curve_v2_0', 'Curve_v2_1', 'Curve_v2_2', 'Curve_v2_3', 'Curve_v2_4', 'Curve_v2_5', 'Curve_v2_6',
           'Curve_v2_7', 'Curve_v2_8', 'Curve_v2_9',
           'Curve_v3_0', 'Curve_v3_1', 'Curve_v3_2', 'Curve_v3_3', 'Curve_v3_4', 'Curve_v3_5', 'Curve_v3_6',
           'Curve_v3_7', 'Curve_v3_8', 'Curve_v3_9']
    df_curve = pd.DataFrame(data=curve, columns=col)
    df_U = pd.DataFrame(data=U, columns=['StrainU_v1', 'StrainU_v2', 'StrainU_v3'])
    df_G = pd.DataFrame(data=G, columns=['Flex_0', 'Flex_1', 'Flex_2', 'Flex_3', 'Flex_4', 'Flex_5', 'Flex_6', 'Flex_7',
                                         'Flex_8', 'Flex_9'])
    return df_curve, df_U, df_G


# ----------- NonModal_features ----------------
def get_low_med_hig_std():  # Obdtains Low, Medium and High. Frequency integrals. Devided by the total integral
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
            f, Gxx_f = welch(df[sensor], nperseg=L // 2, fs=100)

            dx = f[1] - f[0]
            Gxx_low = Gxx_f[f < lim1]
            Gxx_med = Gxx_f[np.logical_and(f > lim1, f < lim2)]
            Gxx_high = Gxx_f[f > lim2]
            a = sqrt(integrate.trapz(Gxx_low, dx=dx))
            b = sqrt(integrate.trapz(Gxx_med, dx=dx))
            c = sqrt(integrate.trapz(Gxx_high, dx=dx))
            low_freq[seg_nr_tot, j] = a / (a + b + c)  # sum(std)=1 might not be recomended
            med_freq[seg_nr_tot, j] = b / (a + b + c)  # compared to sum(var)=1
            high_freq[seg_nr_tot, j] = c / (a + b + c)
    return pd.DataFrame(data=low_freq), pd.DataFrame(data=med_freq), pd.DataFrame(data=high_freq)


def get_pca_res():  # Obtain PCA residual 1 and 18
    def normalize(array):  # normalizes each Sxixi to length 1
        def normalize_2(array, l, col):
            for seg_nr_tot in range(350):
                for i, co in enumerate(col):
                    array[seg_nr_tot, i * l:(i + 1) * l] /= np.linalg.norm(array[seg_nr_tot, i * l:(i + 1) * l])
            return array

        return normalize_2(array, array.shape[1] // 75, np.array(unpickle(column_only=True)))

    def lf_svd(array):  # Use svd to get base case Gxx shapes
        l = array.shape[1] // 75
        col = unpickle(column_only=True)
        u_mat = np.zeros((array.shape[1], 50))
        for i, co in enumerate(col):
            u, s, v = svd(np.transpose(array[0:50, i * l:(i + 1) * l]), compute_uv=True)
            u_mat[i * l:(i + 1) * l, :] = u[:, :50]
        pd.to_pickle(u_mat, 'LF_spectra')
        return u_mat

    def get_Gxx_mat(n_runs=7, n_seg=50, Bins=2, start_seggis=0):  # Obtain the full Gxx as a file
        col = unpickle(column_only=True)
        L = unpickle(n_seg=n_seg)['AL01'].index[-1] + 1
        f, b = welch(unpickle(n_seg=n_seg)['AL01'], nperseg=L // Bins, fs=100)
        l = len(f)
        Gxx_mat = np.zeros([n_runs * n_seg, l * len(col)], dtype='float32')
        pd.to_pickle(Gxx_mat, 'All_Gxx')
        for seggis in range(start_seggis, n_runs * n_seg):
            hold = pd.read_pickle('All_Gxx')[seggis, :]
            r_i, seg_nr = seggis // n_seg, seggis % n_seg
            if hold[0] == 0:
                print(seggis)
                for i, co in enumerate(col):
                    f, Gxx = welch(unpickle(run=r_i, n_seg=n_seg, seg_nr=seg_nr, runbyindex=True)[co],
                                   nperseg=L // Bins, fs=100)
                    hold[i * l:(i + 1) * l] = Gxx * 10 ** 13
            Gxx_mat = pd.read_pickle('All_Gxx')
            Gxx_mat[seggis, :] = hold
            pd.to_pickle(Gxx_mat, 'All_Gxx')
        return

    def feat_svd():  # get the residual values
        get_Gxx_mat()  # This can be commented out if 'All_Gxx' is already produced
        col = np.array(unpickle(column_only=True))
        u_mat = lf_svd(normalize(pd.read_pickle('All_Gxx')))  #
        Gxx = normalize(pd.read_pickle('All_Gxx'))
        l = Gxx.shape[1] // 75
        r = np.zeros((350, len(col), 50))
        for seg_nr_tot in range(350):
            for i, co in enumerate(col):
                for r_i in range(50):
                    r[seg_nr_tot, i, r_i] = -np.dot(u_mat[i * l:(i + 1) * l, r_i], Gxx[seg_nr_tot, i * l:(i + 1) * l])
        return np.square(r)

    r = feat_svd()
    feat1 = 1 - np.sum(r[:, :, :1], axis=-1)
    feat18 = 1 - np.sum(r[:, :, :18], axis=-1)
    return pd.DataFrame(data=feat1), pd.DataFrame(data=feat18)


def add_DMG(df, seg=50):  # Add damage states to
    damage = np.zeros(len(df))
    for i in range(len(damage)):
        run_i = i // seg
        if run_i == 0:
            damage[i] = 0
        elif run_i == 1:
            damage[i] = 1
        elif run_i == 2:
            damage[i] = 2
        elif run_i == 3:
            damage[i] = 3
        elif run_i == 4:
            damage[i] = 4
        elif run_i == 5:
            damage[i] = 5
        elif run_i == 6:
            damage[i] = 6
    df['Damage'] = damage
    return df


# ----------- Obtain Local features ----------------
df1 = get_direct()
df2 = get_yuen()
df3 = get_COMAC()
df4, df5, df6 = get_low_med_hig_std()
df7, df8 = get_pca_res()

# ----------- Add names ----------------
col = [None] * (12 + 12 + 1 + 1 + 1 + 1 + 1 + 1)
col[:12] = ['Mode_' + str(s) for s in range(1, 13)]
col[12:24] = ['Yuen_' + str(s) for s in range(1, 13)]
col[24] = 'COMAC'
col[25] = 'Freq_Low'
col[26] = 'Freq_Med'
col[27] = 'Freq_High'
col[28] = 'PCA_Res_1'
col[29] = 'PCA_Res_18'
# ----------- Create files ----------------
for i, el in enumerate(unpickle(sensor_group='all', column_only='True',time=False)):
    array = np.linspace(i, i + 11 * 75, 12, dtype=int)
    df = pd.concat([df1[array], df2[array], df3[i], df4[i], df5[i], df6[i], df7[i], df8[i]], axis=1, join='outer')
    df.columns = col
    pd.to_pickle(df, 'Freq_output_' + str(50) + '_' + el + '.pkl')
# ----------- Obtain (named) Global features ----------------
df1, df2, df3 = get_MAC()
df4 = get_eigf()
df5, df6, df7 = get_curve_stainU_flexG()
df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1, join='outer')
# ----------- Create file ----------------
pd.to_pickle(add_DMG(df), 'Freq_output_' + str(50) + '_all' + '.pkl')
