import pandas as pd
import numpy as np
from sklearn import preprocessing

### Note that in some cases, such as when importing files, the default inputs for the optional parameters might be slightly different for every user and not as presented.

def unpickle(run='02',sensor_group='all',loc='', start='RUN', end='_100Hz_cut',time=True,runbyindex=False,column_only=False,n_seg = 1, seg_nr = 0, reset_time=False,pos_pad=0,neg_pad=0):
    # reads a pickle-file into a pandas dataframe, including only specified sensors
    # Needs access to acceleration data time series
    #Inputs:
    #    run is a number or string exs '02',2,'13',13
    #    sensor_group is either the sensors requested or a group.
    #    there are four groups. "all"-> all, "globalx" or "Gx" (checks first and last)->Global-x (or z), and global or "g"
    #    start and end is to customize the file types
    #    loc is path, not required if in same folder see exs, nb! must end with \
    #    Recommended to change the default loc in this file if your gonna use it often
    #    column_only means that only the list of sensor names is returned
    #    n_seg is number of segmnets, seg_nr is segment nr
    #    runbyindex is wheather or not obtain the runs by their original "file" name (02,05,11...) or the more convenient indexes (0,1,2...)
    #    pos_pad, neg_pad, is if a an extended segment whith data from the neigbouring segments are wanted to avoid boundary problems
    #returns pandas dataframe
    #exs:
        # df=unpickle(run='02',sensor_group='g')
        # df=unpickle(run=5,sensor_group=['AL01', 'AL02'])
        # df=unpickle(run=2, sensor_group='all',start='MVS_P2_RUN',end='_100Hz',
    #               loc=r'C:\Users\mats-\Documents\Phyton\Skripter\Master\Data\100Hz/')"""
    Local = ['AL01', 'AL02', 'AL03', 'AL04', 'AL05', 'AL06', 'AL07', 'AL08', 'AL09', 'AL10',
             'AL11', 'AL12', 'AL13', 'AL14', 'AL15', 'AL16', 'AL17', 'AL18', 'AL19', 'AL20',
             'AL21', 'AL22', 'AL23', 'AL24', 'AL25', 'AL26', 'AL27', 'AL28', 'AL29', 'AL30',
             'AL31', 'AL32', 'AL33', 'AL34', 'AL35', 'AL36', 'AL37', 'AL38', 'AL39', 'AL40']
    Globalx = [ 'AG01x', 'AG02x', 'AG03x', 'AG04x', 'AG05x', 'AG06x', 'AG07x', 'AG08x', 'AG09x',
                'AG10x', 'AG11x', 'AG12x', 'AG13x', 'AG14x', 'AG15x', 'AG16x', 'AG17x', 'AG18x']
    Globalz = [ 'AG01z', 'AG02z', 'AG03z', 'AG04z', 'AG05z', 'AG06z', 'AG07z', 'AG08z',
                'AG10z', 'AG11z', 'AG12z', 'AG13z', 'AG14z', 'AG15z', 'AG16z', 'AG17z', 'AG18z']
    runs = [2, 5, 11, 23, 26, 29, 32]
    if runbyindex:run=runs[run]
    if not(isinstance(run, str)):
        run=format(run, '02d')
    if isinstance(sensor_group, str):
        sensor_group=[sensor_group]
    if len(sensor_group)==1:
        sensor_group=''.join(sensor_group)
        if sensor_group.lower()=='all':
            list = Local + Globalz + Globalx
        elif sensor_group[0].lower()=='g' and sensor_group[-1].lower()=='x':
            list = Globalx
        elif sensor_group[0].lower()=='g' and sensor_group[-1].lower()=='z':
            list = Globalz
        elif sensor_group[0].lower()=='g':
            list = Globalx+Globalz
        elif sensor_group[0].lower()=='l':
            list = Local
        else:
            list = [sensor_group]
    else:
        list=sensor_group
    if column_only:
        return list
    if time==True:
        list.insert(0, 'timestamp')
    df=pd.read_pickle(loc + start + run + end)[list]
    L=df.index[-1]+1
    df=df[L//n_seg*seg_nr-neg_pad:L//n_seg*(seg_nr+1)+pos_pad].reset_index(drop=True)
    if reset_time:df.timestamp = np.linspace(0, (df.index[-1]) * (df.timestamp[1] - df.timestamp[0]), num=df.index[-1] + 1,endpoint=True)
    return df

#------------

def column_names(name_list, number_of_columns, name_of_data):
    #Creates a namelist for a dataframe if there are many similar features, example AR0, AR1, AR2, ...
    #name_list - data you want to create more names of, can also be an empty list.
    #number_of_columns - how many columns to add
    #name_of_data - New column names, example AR0, AR1, ...
    #Returs name_list with the names for the dataframe.
    if name_of_data == 'AR_Coef_':
        for idx in range(number_of_columns + 1):
            name = str(name_of_data) + str(idx)
            name_list.append(name)
    else:
        for idx in range(number_of_columns):
            name = str(name_of_data) + str(idx)
            name_list.append(name)
    return name_list

def choose_n_split(df, str, n_splits):
    """Splits the dataframe into a chosen number n_splits of columns
        input:
            df = dataframe
            str = column of dataframe to split
            n_splits = number of segments to split into (maximum here is 104)"""
    names = ['A', 'B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
             'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'  , 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ',
             'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ',
             'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP',
             'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD', 'CE', 'CF',
             'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV',
             'CW', 'CX', 'CY', 'CX']
    listen = np.array(df[str])
    length = len(listen) // n_splits
    #Make evenly split data
    splitted_df = pd.DataFrame()
    for i in range (n_splits):
        if len(listen[length*i:length*(i+1)]) != length:
            print('Length of lists does not match, supressing the last set of data')
            break
        else:
            splitted_df[names[i]] = listen[length*i:length*(i+1)]
    if n_splits > len(names):
        print('Not enough names')
    return splitted_df

#--------

def getSensorList(sensor_group='all'):
    """Get list of sensor names from selected group. Same names as in unpickle()"""
    Local = ['AL01', 'AL02', 'AL03', 'AL04', 'AL05', 'AL06', 'AL07', 'AL08', 'AL09', 'AL10',
             'AL11', 'AL12', 'AL13', 'AL14', 'AL15', 'AL16', 'AL17', 'AL18', 'AL19', 'AL20',
             'AL21', 'AL22', 'AL23', 'AL24', 'AL25', 'AL26', 'AL27', 'AL28', 'AL29', 'AL30',
             'AL31', 'AL32', 'AL33', 'AL34', 'AL35', 'AL36', 'AL37', 'AL38', 'AL39', 'AL40']
    Globalx = [ 'AG01x', 'AG02x', 'AG03x', 'AG04x', 'AG05x', 'AG06x', 'AG07x', 'AG08x', 'AG09x',
                'AG10x', 'AG11x', 'AG12x', 'AG13x', 'AG14x', 'AG15x', 'AG16x', 'AG17x', 'AG18x']
    Globalz = [ 'AG01z', 'AG02z', 'AG03z', 'AG04z', 'AG05z', 'AG06z', 'AG07z', 'AG08z',
                'AG10z', 'AG11z', 'AG12z', 'AG13z', 'AG14z', 'AG15z', 'AG16z', 'AG17z', 'AG18z']
    if sensor_group.lower() == 'all':
        list = Local + Globalz + Globalx
    elif sensor_group[0].lower() == 'g' and sensor_group[-1].lower() == 'x':
        list = Globalx
    elif sensor_group[0].lower() == 'g' and sensor_group[-1].lower() == 'z':
        list = Globalz
    elif sensor_group[0].lower() == 'g':
        list = Globalx + Globalz
    elif sensor_group[0].lower() == 'l':
        list = Local
    return list

#-------------

def getFeatureList(n_splits, glob=False, filtered=False):
    """Get names of features.
    input:
        n_splits - number of segments in dataset (50 or 100)
        glob - select between global or local (sensor-dependent) features
        filtered - If true, only sensors not discarded after filter method are returned."""
    if filtered == False:
        loc = 'Features_All'
    else:
        loc = 'Features_All_w_Filtering'
    if glob==False:
        df = pd.read_pickle('Features/%s/n_splits_%i_sensor_AL02.pkl'%(loc,n_splits))
    else:
        df = pd.read_pickle('Features/%s/n_splits_%i_global.pkl'%(loc,n_splits))
    features = df.columns
    features = features.drop('Damage')
    return features

#--------------

def chooseFeatures(df, wantedFeatures):
    """ Excludes features in dataframe that are not requested
        input:
            df - dataframe with features and damage state
            wantedFeatures - features wanted in dataframe
        Output:
            df - updated dataframe
    """
    if (type(wantedFeatures) == pd.core.indexes.base.Index):
        wantedFeatures = wantedFeatures.tolist()
    allFeatures = df.columns
    wantedFeatures.append('Damage')
    for i in allFeatures:
        if i not in wantedFeatures:
            df = df.drop([i], 1)
    return df

#---------

def combineSensors(sensors, n_splits, wantedFeatures):
    """Creates a feature space with all sensors and their features.
        Feature name will be changed into 'FeatureName_SensorName'
        input:
            sensors - sensors to include
            n_splits - dataset with n segments (50 or 100)
            wantedFeatures - Features wanted in feature space"""
    df = pd.DataFrame()
    damage = pd.read_pickle('Features/Features_All/n_splits_' + str(n_splits) + '_sensor_' + sensors[0] + '.pkl')[
        'Damage']
    for sensorN in sensors:
        temp_df = pd.read_pickle('Features/Features_All/n_splits_' + str(n_splits) + '_sensor_' + sensorN + '.pkl')
        temp_df = chooseFeatures(temp_df, wantedFeatures)
        temp_df = temp_df.drop('Damage', axis=1)
        temp_df = temp_df.add_suffix('_' + sensorN)
        df = pd.concat([df, temp_df], axis=1)
    df = pd.concat([df, damage], axis=1)
    return df

#--------

def df_preprossecing(df):
    # z-normalizes each feature in the dataframe
    for column in df:
        if column == 'Damage':
            continue
        else:
            df[column] = preprocessing.scale(df[column])  # Normalisere m = 0, std = 1
    return df
    
#----------
