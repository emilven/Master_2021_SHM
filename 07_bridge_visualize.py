import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from handy import unpickle
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from warnings import filterwarnings
filterwarnings('ignore')

# Plots the bridge. Allows for visualization of damage locations, shaker location, sensor locations and
# change in feature value at every sensors for different states.
# REQUIRES FILES WITH FEATURES AND DAMAGE STATES!

def state2damage(state):
    """Transform state nr to damage nr"""
    if state == 0:
        damageNr = []
    elif state == 1:
        damageNr = [1]
    elif state == 2:
        damageNr = [3]
    elif state == 3:
        damageNr = [7]
    elif state == 4:
        damageNr = [8]
    elif state == 5:
        damageNr = [7, 8]
    elif state == 6:
        damageNr = [9]
    return damageNr

#----------- INPUT: DEFINE WANTED FIGURE ----------------

state1 = 0              # Between 0-6. Should be 0, undamaged
state2 = 1              # Between 0-6
segments = [42,41,40,30,20,10]             # Data set number (Between 0-49)
feature = 'STD'         # Name of feature
enableLocal = 1         # Include local part of bridge (deck)
enableGlobal= 1         # Include truss
enableSensors=1         # Plot sensors
enableShaker= 0         # Plot shaker
damageNr = state2damage(state2) # Plot damages

# ----PRELIMINARY FUNCTIONS: -------------

def localVisualize(x_l,y_l):
    """Plot the deck. Local coordinates as input"""
    for i in range(4):
        plt.plot([1, 11], [y_l[i], y_l[i]], color='black')
    for i in range(len(x_l)):
        if i % 2 == 0:
            plt.plot([x_l[i], x_l[i]], [-0.5, 0.5], color='black')
        else:
            plt.plot([x_l[i], x_l[i]], [-0.5, 0.5], color='black')

def globalVisualize(x_g,y_g_low,y_g_arch):
    """Plot the truss. Global coordinates as input"""
    plt.plot(x_g, y_g_arch, color='black')
    plt.plot(x_g, -y_g_arch, color='black')
    plt.plot([1, 11], [1, 1], color='black')
    plt.plot([1, 11], [-1, -1], color='black')
    for i in range(len(x_g)):
        plt.plot([x_g[i], x_g[i]], [y_g_low[i], y_g_arch[i]], color='black')
        plt.plot([x_g[i], x_g[i]], [-y_g_low[i], -y_g_arch[i]], color='black')
        plt.plot([x_g[i], x_g[i]], [y_g_low[i], -y_g_low[i]], color='black', linestyle='--')

def lateralBracingVisualize(x_g,y_g):
    """Plot the lateral bracing. Global coordinates as input"""
    for i in range(len(x_g)-1):
        plt.plot([x_g[i],x_g[i+1]],[y_g[i],-y_g[i+1]], color='black',linestyle='--')
        plt.plot([x_g[i],x_g[i+1]],[-y_g[i],y_g[i+1]], color='black',linestyle='--')

def globalSensorPos(x_g,y_g_low,y_g_arch):
    """Define the coordinates of the global sensors. Global coordinates as input"""
    sens_x_g = np.append(x_g[1:10], x_g[1:10])
    sens_y_g = np.array([])
    for i in range(len(sens_x_g)):
        if i < len(sens_x_g) // 2:
            if i % 2 == 0:
                sens_y_g = np.append(sens_y_g, y_g_low[i])
            else:
                sens_y_g = np.append(sens_y_g, y_g_arch[i + 1])
        else:
            if i % 2 == 0:
                sens_y_g = np.append(sens_y_g, -y_g_low[i - len(sens_x_g) // 2])
            else:
                sens_y_g = np.append(sens_y_g, -y_g_arch[i + 1 - len(sens_x_g) // 2])
    return sens_x_g,sens_y_g

def damageLoc():
    """Define the coordinates of the damages"""
    x_d1 = x_l[8]
    x_d2 = x_l[8]
    x_d3 = np.array([x_l[5], x_l[5]])
    x_d4 = np.array([x_l[7], x_l[7]])
    x_d5 = np.array([x_d1, x_d4])
    x_d6 = np.array([x_d2, x_d4])
    x_d7 = x_g[2]
    x_d8 = x_g[5]
    x_d9 = np.array([x_l[12], np.array([x_l[12], x_l[12]])])
    x_damage = np.array([x_d1, x_d2, x_d3, x_d4, x_d5, x_d6, x_d7, x_d8, x_d9])
    y_d1 = y_l[1]
    y_d2 = y_l[0]
    y_d3 = np.array([y_l[1], y_l[2]])
    y_d4 = np.array([y_l[1], y_l[2]])
    y_d5 = np.array([y_d1, y_d4])
    y_d6 = np.array([y_d2, y_d4])
    y_d7 =-y_g_low[2]
    y_d8 = y_g_low[5]
    y_d9 = np.array([(y_l[0] - y_g_low[7]) / 2, np.array([-y_g_low[7], y_g_low[7]])])
    y_damage = np.array([y_d1, y_d2, y_d3, y_d4, y_d5, y_d6, y_d7, y_d8, y_d9])
    return x_damage,y_damage

def damageVisualize(damageNr,x_damage,y_damage):
    """Plot damages. damageNr ranges from 1-9"""
    damageNr = [i-1 for i in damageNr]
    for i in (damageNr):
        if isinstance(x_damage[i], np.ndarray):
            if isinstance(x_damage[i][1], np.ndarray):
                damageViz = plt.scatter(x_damage[i][0], y_damage[i][0], color='crimson', marker='x', s=250,zorder=15)
                plt.plot(x_damage[i][1], y_damage[i][1], color='red',zorder=15)
            else:
                damageViz = plt.scatter(x_damage[i], y_damage[i], color='crimson', marker='x', s=250,zorder=15)
                plt.plot(x_damage[i], y_damage[i], color='red')
        else:
            damageViz = plt.scatter(x_damage[i], y_damage[i], color='crimson', marker='x', s=250,zorder=15)
        if i == damageNr[0]:
            damageViz.set_label('damage')

def featureSelect(splitNr,sensors,state,feature):
    """Select feature. Input: segment Nr, sensors to evaluate, state to evaluate, feature to evaluate"""
    n_splits = 50
    featureVal = []
    for sensorN in sensors:
        df = pd.read_pickle('Features/Features_All/n_splits_' + str(n_splits) + '_sensor_' + sensorN + '.pkl')
        dfMod = df.loc[df['Damage'] == state]
        dfMod = dfMod[feature]
        dfMod = dfMod.iloc[splitNr]
        featureVal.append(dfMod)
    featureVal = np.array(featureVal)
    return featureVal

def localDiff(state1,state2,feature,dataSet, normMethod=1):
    """Calculate the relative difference between 2 states, local sensors.
        Input: States to compare, feature to compare, segment to compare, normalization method."""
    sensors = unpickle(run='02', sensor_group='l', time=False).columns
    feat1 = featureSelect(dataSet, sensors, state1, feature)
    feat2 = featureSelect(dataSet, sensors, state2, feature)
    if normMethod == 1:
        diffNorm = abs(feat2-feat1)
    elif normMethod == 2:
        diffNorm = abs(feat2 / feat1 - 1)
    return diffNorm

def globalDiff(state1,state2,feature,dataSet, normMethod=1):
    """Calculate the relative difference between 2 states, global sensors.
        Input: States to compare, feature to compare, segment to compare, normalization method."""
    sensorsX = unpickle(run='02', sensor_group='gx', time=False).columns
    sensorsZ = unpickle(run='02', sensor_group='gz', time=False).columns
    feat1x = featureSelect(dataSet, sensorsX,state1, feature)
    feat1z = featureSelect(dataSet, sensorsZ,state1, feature)
    feat2x = featureSelect(dataSet, sensorsX,state2, feature)
    feat2z = featureSelect(dataSet, sensorsZ,state2, feature)
    if normMethod == 1:
        diffNormx = abs(feat2x-feat1x)
        diffNormz = abs(feat2z-feat1z)
    elif normMethod == 2:
        diffNormx = abs(feat2x / feat1x-1)
        diffNormz = abs(feat2z / feat1z-1)
    return diffNormx, diffNormz

def plotGlobalArrows(diffx_g,diffz_g,x_g,y_g,minVal,maxVal):
    """Plot global sensors as arrows and targets.
        Input: differance between sensors, global coordinates, limit values"""
    cNormx = colors.Normalize(vmin=minVal, vmax=maxVal)
    scalarMapx = cmx.ScalarMappable(norm=cNormx, cmap='CMRmap_r')
    cNormy = colors.Normalize(vmin=minVal, vmax=maxVal)
    scalarMapy = cmx.ScalarMappable(norm=cNormy, cmap='CMRmap_r')
    count = 0
    for i in range(18):
        colorValx = scalarMapx.to_rgba(diffx_g[i])
        if i < 9:
            plt.arrow(x_g[i], y_g[i], 0, 0.8, width=0.05, color=colorValx,zorder=11)
        else:
            plt.arrow(x_g[i], y_g[i], 0, -0.8, width=0.05, color=colorValx,zorder=11)
        if i != 8:
            colorValz = scalarMapy.to_rgba(diffz_g[i - count])
            plt.scatter(x_g[i], y_g[i], s=250, facecolor='none', edgecolors=colorValz, linewidths=2, zorder=12)
            plt.scatter(x_g[i], y_g[i], s=40, color=colorValz, zorder=12)
        else:
            count = 1

#--------------- PLOT VISUALIZATION --------------

# Plot one or several plots
for dataSet in segments:
    # ESTABLISH FIGURE WINDOW AND TITLE
    title = 'Feature: ' + feature + '  |  State 0 v State ' + str(state2) + '  |  Segment #' + str(dataSet)
    if enableGlobal == 1:
        plt.figure(figsize=(13,5))
    else:
        plt.figure(figsize=(13,2))
        plt.ylim([-1,1])

    # ESTABLISH VALUE LIMITS
    minVal = 0
    if enableGlobal == 1:
        diffNorm_g_x, diffNorm_g_z = globalDiff(state1, state2, feature, dataSet)
        maxx, maxy = max(diffNorm_g_x), max(diffNorm_g_z)
        maxVal = max(maxx, maxy)
        if enableLocal == 1:
            diffNorm_l = localDiff(state1, state2, feature, dataSet)
            maxl =max(diffNorm_l)
            maxVal = max(maxx,maxy,maxl)
    else:
        diffNorm_l = localDiff(state1, state2, feature, dataSet)
        maxVal = max(diffNorm_l)

    # DEFINE LOCAL POINTS
    x_l = np.arange(1,11.5,0.5)
    y_l = np.array([-0.5,-0.3,0.3,0.5])

    if enableLocal == 1:
        # PLOT LOCAL FRAME
        localVisualize(x_l,y_l)

    # DEFINE LOCAL SENSORS
    sens_x_l = np.arange(1.5,11.5,1)
    sens_y_l = y_l
    xv_l,yv_l=np.meshgrid(sens_x_l,sens_y_l)
    xv_l, yv_l = np.transpose(xv_l), np.transpose(yv_l)

    if (enableLocal == 1) & (enableSensors == 1):
        # PLOT LOCAL SENSORS
        plt.scatter(xv_l,yv_l, c=diffNorm_l, cmap="CMRmap_r", vmin=minVal,vmax=maxVal, s=30,zorder=10)
        # CREATE COLORBAR
        if enableGlobal == 1:
            plt.colorbar(fraction=0.02)
        else:
            plt.colorbar(fraction=0.02,aspect=10)

    # DEFINE GLOBAL POINTS
    x_g = np.arange(1,12,1)
    y_g_arch = np.array([])
    y_g_low = np.full([11,1],1)
    for i in range(len(x_g)):
        temp = 2 + 0.6*i - 0.06*i**2
        y_g_arch = np.append(y_g_arch,temp)
    if enableGlobal==1:
        # PLOT GLOBAL FRAME
        globalVisualize(x_g,y_g_low,y_g_arch)
        # lateralBracingVisualize(x_g,y_g_low) #If desired

    # DEFINE GLOBAL SENSORS
    sens_x_g,sens_y_g = globalSensorPos(x_g,y_g_low,y_g_arch)

    if (enableGlobal==1) & (enableSensors == 1):
        # PLOT GLOBAL SENSORS
        plotGlobalArrows(diffNorm_g_x, diffNorm_g_z, sens_x_g, sens_y_g, minVal, maxVal)
        if enableLocal == 0:
            # CREATE COLORBAR
            plt.colorbar(fraction=0.02)

    if enableShaker == 1:
        # PLOT THE SHAKER
        plt.scatter(6,0,marker='s',s=100,zorder=9,label='Shaker')

    if damageNr != [0]:
        # PLOT THE DAMAGES
        x_damage, y_damage = damageLoc()
        damageVisualize(damageNr,x_damage,y_damage)

    # FINISH FIGURE
    plt.title(title, size=15)
    plt.legend()
    plt.yticks([])
    plt.xticks([])
