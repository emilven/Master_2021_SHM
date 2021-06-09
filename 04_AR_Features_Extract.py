import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from scipy.stats import kurtosis, skew
from warnings import simplefilter
from handy import unpickle
from handy import column_names
from handy import choose_n_split
simplefilter(action='ignore', category=FutureWarning)
from warnings import filterwarnings
filterwarnings('ignore')

def AR_basedata(baseline, p):
  ''' Extracts (p+1) AR coefficients of a time series and the mean and standard deviation of its residual
    Input:
      baseline: dataframe of a time series
      p: Order of the AR model to create'''
    ar_coef_list = []
    ar_model = AutoReg(baseline, p)
    model = ar_model.fit()
    mean_b = np.mean(model.resid)
    std_b = np.std(model.resid)
    ar_coef_list.append(model.params)
    return ar_coef_list, mean_b, std_b

def AR_testdata(testdata, p, ar_coef_b, mean_b, std_b):
      ''' Creates an AR model of a time series, extracts the AR coefficients and creates statistical features.
    Input:
      testdata: Timeseries to create model of and extract features
      p: Order of the AR model to create
      ar_coef_b: AR coefficients to compare to the model created from testdata
      mean_b: Mean of the undamaged state
      std_b: Standard deviation of the undamaged state'''
    ar_model = AutoReg(testdata, p)
    ar_model_fit_t = ar_model.fit()
    ar_coef_t = ar_model_fit_t.params
    ar_coef_list = list(ar_coef_t)

    #Predicting data using undamaged AR coefficients
    ar_model_fit = AutoReg.predict(ar_model, list(ar_coef_b[0]))
    residual_test = testdata[p::] - ar_model_fit
    
    #Features to extract
    mean_t = np.mean(residual_test)
    std_t = np.std(residual_test, ddof=1)
    skewness_t = skew(residual_test)
    kurtosis_t = kurtosis(residual_test)
    rms = np.sqrt(np.mean(residual_test ** 2))
    correlation = acf(residual_test, fft=False, nlags=1)
    pdf = (residual_test - mean_b) / std_b
    amplitude = abs(residual_test).quantile(0.98)
    outlier_r = 0
    outlier_l = 0
    for element in pdf:
        if element < -1.96:
            outlier_l += 1
        elif element > 1.96:
            outlier_r += 1

    return mean_t, std_t, skewness_t, kurtosis_t, outlier_l, outlier_r, amplitude, rms, correlation[1], ar_coef_list

ar_order = 10
n_splits = 100
damage_list = [0, 1, 2, 3, 4, 5, 6]
feature_names = ['AR_Mean', 'AR_STD', 'AR_Skewness', 'AR_Kurtosis', 'AR_Outlier_L', 'AR_Outlier_R', 'AR_Peak', 'AR_RMS', 'AR_Auto_Corr']
feature_names = column_names(feature_names, ar_order, 'AR_Coef_')
feature_names.append('Damage')
sensor_names = ['AL01', 'AL02', 'AL03', 'AL04', 'AL05', 'AL06', 'AL07',
              'AL08', 'AL09', 'AL10', 'AL11', 'AL12', 'AL13', 'AL14', 'AL15', 'AL16',
              'AL17', 'AL18', 'AL19', 'AL20', 'AL21', 'AL22', 'AL23', 'AL24', 'AL25',
              'AL26', 'AL27', 'AL28', 'AL29', 'AL30', 'AL31', 'AL32', 'AL33', 'AL34',
              'AL35', 'AL36', 'AL37', 'AL38', 'AL39', 'AL40', 'AG01x',
                'AG02x', 'AG03x', 'AG04x', 'AG05x', 'AG06x', 'AG07x', 'AG08x', 'AG09x',
                'AG10x', 'AG11x', 'AG12x', 'AG13x', 'AG14x', 'AG15x', 'AG16x', 'AG17x', 'AG18x', 'AG01z',
                'AG02z', 'AG03z', 'AG04z', 'AG05z', 'AG06z', 'AG07z', 'AG08z',
                'AG10z', 'AG11z', 'AG12z', 'AG13z', 'AG14z', 'AG15z', 'AG16z', 'AG17z', 'AG18z']
sensor_list = ['sensor_base', 'sensor_05', 'sensor_11', 'sensor_23', 'sensor_26', 'sensor_29', 'sensor_32']

baseline = unpickle(run='02')
df05 = unpickle(run='05')
df11 = unpickle(run='11')
df23 = unpickle(run='23')
df26 = unpickle(run='26')
df29 = unpickle(run='29')
df32 = unpickle(run='32')

for sensor in sensor_names:
    sensor_base = choose_n_split(baseline, sensor, n_splits)
    sensor_05 = choose_n_split(df05, sensor, n_splits)
    sensor_11 = choose_n_split(df11, sensor, n_splits)
    sensor_23 = choose_n_split(df23, sensor, n_splits)
    sensor_26 = choose_n_split(df26, sensor, n_splits)
    sensor_29 = choose_n_split(df29, sensor, n_splits)
    sensor_32 = choose_n_split(df32, sensor, n_splits)
    ar_coef_b, mean_b, std_b = AR_basedata(baseline[sensor], ar_order)
    res = []
    for idx, element in enumerate(sensor_list):
        for column in eval(element):
            mean_t, sts_t, skewness_t, kurtosis_res, outlier_l, outlier_r, amplitude, rms, correlation, ar_coef_list = AR_testdata(eval(element)[column], ar_order, ar_coef_b, mean_b, std_b)
            results = list([mean_t, sts_t, skewness_t, kurtosis_res, outlier_l, outlier_r, amplitude, rms, correlation] +  ar_coef_list)
            results.append(damage_list[idx])
            res.append(results)


    features = pd.DataFrame(res, columns=feature_names)
    save_path = str('AR_results/AR_output_' + str(n_splits) + str(sensor) + '.pkl')
    features.to_pickle(save_path)



