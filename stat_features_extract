import pandas as pd
import numpy as np
import handy
from scipy import stats as stat
from statsmodels.tsa.stattools import acf

def auto_corr_function(splitted_df):
    """Calculate autocorrelation with timelag equal to one time step"""
    auto_corr = []
    for col in splitted_df.columns:
        split = splitted_df[col]
        auto_corr_temp = acf(split, fft=False, nlags=1)[1]
        auto_corr.append(auto_corr_temp)
    return auto_corr

def getOutliersZ(df,mean0,std0):
    """Find the outliers with novelty detection using deviation statistics (z-score)"""
    outlierR = []
    outlierL = []
    for sensorN in df.columns:
        df_temp = df[sensorN]
        pdf_t = (df_temp - mean0) / std0
        outlier_r = 0
        outlier_l = 0
        for element in pdf_t:
            if element < -1.96:
                outlier_l += 1
            elif element > 1.96:
                outlier_r += 1
        outlierL.append(outlier_l)
        outlierR.append(outlier_r)
    outlierR = pd.DataFrame(outlierR, df.columns)
    outlierL = pd.DataFrame(outlierL, df.columns)
    return outlierR, outlierL

#--------- MAKE FILES FOR EACH SENSOR WITH THE STATISTICAL FEATURES USED IN THESIS ------------

# -------------- Input ------------
n_splits = 50                                               # 50 or 100
runFileNr = ["02", "05", "11", "23", "26", "29", "32"]      # All files used in thesis
sensorGroup = 'all'                                         # Sensors to extract features for
state = [0, 1, 2, 3, 4, 5, 6]                               # Include all states

#---------------------------------------

# Calculate base case observations
df = handy.unpickle(run="02", sensor_group='all', time=False)
mean0 = np.mean(df)
std0 = np.std(df)

# Get name of sensors
sensors = df.columns

# Create file for each sensor
for sensorN in sensors:
    stat_coeff = pd.DataFrame()
    stat_coeff_temp = pd.DataFrame()
    # Get features for every state
    for run in runFileNr:
        df = handy.unpickle(run=run,sensor_group = sensorGroup,time=False)
        #Split into segments
        splitted_df = handy.choose_n_split(df,sensorN,n_splits)

        #Calculate feature values
        mean_data = np.mean(splitted_df)
        med_data = np.median(splitted_df, 0)
        peak_data = abs(abs(splitted_df).quantile(0.98) - abs(mean_data))
        std_data = np.std(splitted_df)
        skew_data = stat.skew(splitted_df)  # /std_data**3      # Include last part to make unitless
        kurt_data = stat.kurtosis(splitted_df)#/std_data**4     # Include last part to make unitless
        rms_data = np.sqrt(np.mean(splitted_df ** 2))
        #crest_fac = peak_data / rms_data                       # Not included
        #k_factor = peak_data * rms_data                        # Not included
        [outlierR,outlierL] = getOutliersZ(splitted_df,mean0[sensorN],std0[sensorN])
        auto_corr = auto_corr_function(splitted_df)

        stat_coeff_temp['Mean'] = mean_data
        stat_coeff_temp['Median'] = med_data
        stat_coeff_temp['STD'] = std_data
        stat_coeff_temp['RMS'] = rms_data
        stat_coeff_temp['Peak'] = peak_data
        stat_coeff_temp['Skewness'] = skew_data
        stat_coeff_temp['Kurtosis'] = kurt_data
        stat_coeff_temp['Outlier_R'] = outlierR
        stat_coeff_temp['Outlier_L'] = outlierL
        stat_coeff_temp['Auto_Corr'] = auto_corr
        stat_coeff_temp['Damage'] = state[runFileNr.index(run)]

        # Include features from segment
        stat_coeff = stat_coeff.append(stat_coeff_temp)
    stat_coeff.reset_index(drop=True, inplace=True)

    #Save sensor features to a file
    save_path = str('./statData/Stat_output_'+str(n_splits)+'_'+sensorN+'.pkl')
    stat_coeff.to_pickle(save_path)
