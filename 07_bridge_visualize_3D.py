import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from handy import unpickle
from warnings import simplefilter
# ignore all future warnings:
simplefilter(action='ignore', category=FutureWarning)
from warnings import filterwarnings
filterwarnings('ignore')

# User guide:
    # Similar to visualize_bridge, but plots a 3D-model of the bridge and visualized the relative change between
    # features as displacement.
    # Look at that guide for instructions.

#INPUT: DEFINE WANTED FIGURE
state1 = 0              # Between 0-6. Should be 0, undamaged
state2 = 4              # Between 0-6.
feature = 'Mode_12'   # Name of feature. NB! Must exist in statFeatureBase
scale = .6              # Scale of displacement
c = 'red'               # Colour to plot displaced bridge in
dataSet = 40            # Data set nr from 0-49

enableLocal = 0         # Include local part of bridge (span)
enableGlobal= 1         # Include truss

enableSensors=1         # Enable/disable sensors
enableShaker= 0         # Include shaker
damageNr = [8]           # Include damages. See below for info

# - State-to-damage-list:
    # State0: Undamaged condition, shaker noise mode, shaker position P2, vertical direction.
    # State1: Damage D1, -"-
    # State2: Damage D3 -"-
    # State3: Damage D7 -"-
    # State4: Damage D8 -"-
    # State5: Damage D7+D8 -"-
    # State6: Damage D9 -"-
#-----

# Preliminary Functions:
def localVisualize(x_l,y_l,ax):
    """Plot the deck. Local coordinates as input"""
    for i in range(4):
        ax.plot3D([1, 11], [y_l[i], y_l[i]], [0,0], color='grey')
    for i in range(len(x_l)):
        if i % 2 == 0:
            ax.plot3D([x_l[i], x_l[i]], [-0.5, 0.5], [0,0], color='grey')
        else:
            ax.plot3D([x_l[i], x_l[i]], [-0.5, 0.5], [0,0], color='grey')

def localVisualize_withSensors(sens_x_l,sens_y_l,sens_z_l,ax, c='red'):
    """Plot the deck with sensors visible"""
    x = np.insert(sens_x_l,0,[1,1,1,1])
    x = np.append(x,[11,11,11,11])
    y = np.insert(sens_y_l,0,[-0.5,-0.3,0.3,0.5])
    y = np.append(y,[-0.5,-0.3,0.3,0.5])
    z = np.insert(sens_z_l,0,[0,0,0,0])
    z = np.append(z,[0,0,0,0])
    for i in range(12):
        for j in range(3):
            k = i*4+j
            ax.plot3D([x[k],x[k+1]], [y[k],y[k+1]], [z[k],z[k+1]], color = c)
    for i in range(4):
        for j in range(11):
            k = i + j*4
            ax.plot3D([x[k],x[k+4]], [y[k],y[k+4]], [z[k],z[k+4]], color = c)

def globalVisualize(x_g,y_g,z_g_low,z_g_arch,ax):
    """Plot the truss. Global coordinates as input"""
    ax.plot3D(x_g, y_g, z_g_arch, color='grey')
    ax.plot3D(x_g,-y_g, z_g_arch, color='grey')
    ax.plot3D([1, 11], [y_g[0], y_g[0]], [0,0], color='grey')
    ax.plot3D([1, 11], [-y_g[0], -y_g[0]], [0,0], color='grey')
    for i in range(len(x_g)):
        ax.plot3D([x_g[i], x_g[i]], [ y_g[i], y_g[i]], [z_g_low[i], z_g_arch[i]], color='grey')
        ax.plot3D([x_g[i], x_g[i]], [-y_g[i],-y_g[i]], [z_g_low[i], z_g_arch[i]], color='grey')

def globalVisualize_withSensors(sens_x_g,sens_y_g,sens_z_g,z_g_arch,ax,c='red'):
    """Plot the truss with sensors visible"""
    x_bc = np.array([1,11])
    y_bc = np.array([sens_y_g[0],-sens_y_g[0]])
    z_upper_bc = np.array([z_g_arch[0],z_g_arch[10]])
    z_lower_bc = np.array([0,0])

    x_l = sens_x_g[0:9]
    x_r = (sens_x_g[9:])
    y_l = sens_y_g[0:9]
    y_r = (sens_y_g[9:])
    z_l = sens_z_g[0:9]
    z_r = (sens_z_g[9:])

    ax.plot3D([x_bc[0], x_bc[0]], [y_bc[0], y_bc[0]], [z_upper_bc[0], z_lower_bc[0]], color=c)
    ax.plot3D([x_bc[0], x_bc[0]], [y_bc[1], y_bc[1]], [z_upper_bc[1], z_lower_bc[1]], color=c)
    ax.plot3D([x_bc[1], x_bc[1]], [y_bc[0], y_bc[0]], [z_upper_bc[0], z_lower_bc[0]], color=c)
    ax.plot3D([x_bc[1], x_bc[1]], [y_bc[1], y_bc[1]], [z_upper_bc[1], z_lower_bc[1]], color=c)

    for i in range(10):
        if i == 0:
            temp_l = np.array([(x_l[i+1]+x_bc[0])/2,y_l[i],(z_l[i+1]+z_upper_bc[0])/2])
            temp_r = np.array([(x_r[i+1]+x_bc[0])/2,y_r[i],(z_r[i+1]+z_lower_bc[0])/2])
            temp_l_old = np.array([x_bc[0], y_bc[0], z_lower_bc[0]])
            temp_r_old = np.array([x_bc[0], y_bc[1], z_upper_bc[0]])
            ax.plot3D([temp_l[0], x_l[i]], [temp_l[1],y_l[i]], [temp_l[2],z_l[i]], color=c)
            ax.plot3D([temp_r[0], x_r[i]], [temp_r[1],y_r[i]], [temp_r[2],z_r[i]], color=c)
            ax.plot3D([temp_l_old[0], x_l[i]], [temp_l_old[1], y_l[i]], [temp_l_old[2], z_l[i]], color=c)
            ax.plot3D([temp_r_old[0], x_r[i]], [temp_r_old[1], y_r[i]], [temp_r_old[2], z_r[i]], color=c)
            ax.plot3D([x_bc[0], temp_l[0]], [y_bc[0], temp_l[1]], [z_upper_bc[0], temp_l[2]], color=c)
            ax.plot3D([x_bc[0], temp_r[0]], [y_bc[1], temp_r[1]], [z_lower_bc[0], temp_r[2]], color=c)
        elif i<8:
            temp_l_old = temp_l
            temp_r_old = temp_r
            temp_l = np.array([(x_l[i+1]+x_l[i-1])/2, (y_l[i+1]+y_l[i-1])/2, (z_l[i+1]+z_l[i-1])/2])
            temp_r = np.array([(x_r[i+1]+x_r[i-1])/2, (y_r[i+1]+y_r[i-1])/2, (z_r[i+1]+z_r[i-1])/2])
            ax.plot3D([temp_l[0], x_l[i]], [temp_l[1], y_l[i]], [temp_l[2], z_l[i]], color=c)
            ax.plot3D([temp_r[0], x_r[i]], [temp_r[1], y_r[i]], [temp_r[2], z_r[i]], color=c)
            ax.plot3D([temp_l_old[0],x_l[i]], [temp_l_old[1],y_l[i]], [temp_l_old[2],z_l[i]], color=c)
            ax.plot3D([temp_r_old[0],x_r[i]], [temp_r_old[1],y_r[i]], [temp_r_old[2],z_r[i]], color=c)
            ax.plot3D([x_l[i-1],temp_l[0]], [y_l[i-1],temp_l[1]], [z_l[i-1],temp_l[2]], color=c)
            ax.plot3D([x_r[i-1],temp_r[0]], [y_r[i-1],temp_r[1]], [z_r[i-1],temp_r[2]], color=c)
        elif i == 8:
            temp_l_old = temp_l
            temp_r_old = temp_r
            temp_l = np.array([(x_l[i-1]+x_bc[1])/2, y_l[i], (z_l[i-1]+z_upper_bc[0])/2])
            temp_r = np.array([(x_r[i-1]+x_bc[1])/2, y_r[i], (z_r[i-1]+z_lower_bc[0])/2])
            ax.plot3D([temp_l[0], x_l[i]], [temp_l[1], y_l[i]], [temp_l[2], z_l[i]], color=c)
            ax.plot3D([temp_r[0], x_r[i]], [temp_r[1], y_r[i]], [temp_r[2], z_r[i]], color=c)
            ax.plot3D([temp_l_old[0], x_l[i]], [temp_l_old[1], y_l[i]], [temp_l_old[2], z_l[i]], color=c)
            ax.plot3D([temp_r_old[0], x_r[i]], [temp_r_old[1], y_r[i]], [temp_r_old[2], z_r[i]], color=c)
            ax.plot3D([x_l[i-1],temp_l[0]], [y_l[i-1],temp_l[1]], [z_l[i-1],temp_l[2]], color=c)
            ax.plot3D([x_r[i-1],temp_r[0]], [y_r[i-1],temp_r[1]], [z_r[i-1],temp_r[2]], color=c)
        else:
            temp_l_old = temp_l
            temp_r_old = temp_r
            ax.plot3D([temp_l_old[0], x_bc[1]], [temp_l_old[1], y_bc[0]], [temp_l_old[2], z_upper_bc[1]], color=c)
            ax.plot3D([temp_r_old[0], x_bc[1]], [temp_r_old[1], y_bc[1]], [temp_r_old[2], z_lower_bc[1]], color=c)
            ax.plot3D([x_l[i - 1], x_bc[1]], [y_l[i - 1], y_bc[0]], [z_l[i - 1], z_lower_bc[1]], color=c)
            ax.plot3D([x_r[i - 1], x_bc[1]], [y_r[i - 1], y_bc[1]], [z_r[i - 1], z_upper_bc[1]], color=c)

def globalSensorPos(x_g,y_g,z_g_low,z_g_arch):
    """Define the coordinates of the global sensors. Global coordinates as input"""
    sens_x_g = np.append(x_g[1:10], x_g[1:10])
    sens_y_g = np.append(y_g[1:10], -y_g[1:10])
    sens_z_g = np.array([])
    for i in range(len(sens_x_g)):
        if i < len(sens_x_g) // 2:
            if i % 2 == 0:
                sens_z_g = np.append(sens_z_g, z_g_low[i])
            else:
                sens_z_g = np.append(sens_z_g, z_g_arch[i + 1])
        else:
            if i % 2 == 0:
                sens_z_g = np.append(sens_z_g, z_g_low[i - len(sens_x_g) // 2])
            else:
                sens_z_g = np.append(sens_z_g, z_g_arch[i + 1 - len(sens_x_g) // 2])
    return sens_x_g, sens_y_g, sens_z_g

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
    y_d7 =-z_g_low[2]
    y_d8 = z_g_low[5]
    y_d9 = np.array([(y_l[0] - z_g_low[7]) / 2, np.array([-z_g_low[7], z_g_low[7]])])
    y_damage = np.array([y_d1, y_d2, y_d3, y_d4, y_d5, y_d6, y_d7, y_d8, y_d9])
    return x_damage,y_damage

def damageVisualize(damageNr,x_damage,y_damage):
    """Plot damages. damageNr ranges from 1-9"""
    damageNr = [i-1 for i in damageNr]
    for i in (damageNr):
        if isinstance(x_damage[i], np.ndarray):
            if isinstance(x_damage[i][1], np.ndarray):
                x,y = x_damage[i][0] , y_damage[i][0]
                damageViz = plt.scatter(x, y, color='crimson', marker='x', s=150,zorder=15)
                plt.plot(x_damage[i][1], y_damage[i][1], color='red',zorder=15)
            else:
                x,y = x_damage[i] , y_damage[i]
                damageViz = plt.scatter(x, y, color='crimson', marker='x', s=150,zorder=15)
                plt.plot(x, y, color='red')
                x,y = (x[0]+x[1])/2 , (y[0]+y[1])/2
        else:
            x,y = x_damage[i] , y_damage[i]
            damageViz = plt.scatter(x, y, color='crimson', marker='x', s=150,zorder=15)
        if i == damageNr[0]:
            damageViz.set_label('damage')
        txt = plt.annotate('D' + str(i + 1), (x+0.1, y), color='crimson',fontsize = 12)
        txt.set_bbox(dict(facecolor='white',alpha=0.5,edgecolor = 'none'))

def featureSelect(splitNr,sensors,state,feature):
    """Select feature. Input: segment Nr, sensors to evaluate, state to evaluate, feature to evaluate"""
    n_splits = 50
    featureVal = []
    for sensorN in sensors:
        df = pd.read_pickle('n_splits_'+str(n_splits)+'_sensor_'+sensorN+'.pkl')
        dfMod = df.loc[df['Damage'] == state]
        dfMod = dfMod[feature]
        dfMod = dfMod.iloc[splitNr]
        featureVal.append(dfMod)
    featureVal = np.array(featureVal)
    return featureVal

def sensorPush(state,feature,dataSet):
    """Displace sensors from their initial position related to the change of feature value."""
    sensors = unpickle(run='02', sensor_group='all', time=False).columns
    featVal = featureSelect(dataSet, sensors, state, feature)
    push_l = featVal[0:40]
    push_g_y = featVal[40:57]
    push_g_y = np.insert(push_g_y,8,0)
    push_g_y = np.append(-push_g_y[0:9],push_g_y[9:])
    push_g_z = featVal[57:]
    return push_l, push_g_y, push_g_z

def axisEqual3D(ax):
    """Remove axis lines from plot"""
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

###########################
plt.figure(figsize=[8,5.5])
title = 'Feature: ' + feature + '  |  State 0 v State ' + str(state2) + '  |  Segment #' + str(dataSet)
#ENABLE FIGURE
ax = plt.gca(projection='3d')
ax.axis('off')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.subplots_adjust(bottom=0.0)

#DEFINE MODE SHAPE VALUES
push_l_1, push_g_y_1, push_g_z_1 = sensorPush(state1,feature,dataSet)
push_l_2, push_g_y_2, push_g_z_2 = sensorPush(state2,feature,dataSet)
push_l, push_g_y, push_g_z = push_l_2-push_l_1, push_g_y_2-push_g_y_1, push_g_z_2-push_g_z_1
push_max = max(abs(push_l))
push_l, push_g_y, push_g_z = push_l/push_max*scale, push_g_y/push_max*scale, push_g_z/push_max*scale

#DEFINE LOCAL POINTS
x_l = np.arange(1,11.5,0.5)
y_l = np.array([-0.5,-0.3,0.3,0.5])
z_l = np.array([0,0,0,0])

if enableLocal == 1:
    #PLOT LOCAL FRAME
    localVisualize(x_l,y_l,ax)

#DEFINE LOCAL SENSORS
sens_x_l = np.arange(1.5,11.5,1)
sens_y_l = y_l
xv_l, yv_l = np.meshgrid(sens_x_l,sens_y_l)
xv_l = np.concatenate(np.stack(xv_l,axis=1),axis=0)
yv_l = np.concatenate(np.stack(yv_l,axis=1),axis=0)
zv_l = np.full([1,40],0)
zv_l_push = zv_l + push_l

if (enableLocal == 1) & (enableSensors == 1):
    #PLOT LOCAL SENSORS
    ax.scatter3D(xv_l,yv_l,zv_l, color ='blue', s=2, zorder=10)
    ax.scatter3D(xv_l,yv_l,zv_l_push, color =c, s=10, zorder=10)
    localVisualize_withSensors(xv_l, yv_l, zv_l_push, ax, c)

#DEFINE GLOBAL POINTS
x_g = np.arange(1,12,1)
y_g = np.full([11,1],0.8)
z_g_arch = np.array([])
z_g_low = np.full([11,1],0)
for i in range(len(x_g)):
    temp = 1 + 0.6*i - 0.06*i**2
    z_g_arch = np.append(z_g_arch,temp)
if enableGlobal==1:
    #PLOT GLOBAL FRAME
    globalVisualize(x_g,y_g,z_g_low,z_g_arch,ax)

#DEFINE GLOBAL SENSORS
sens_x_g, sens_y_g, sens_z_g = globalSensorPos(x_g, y_g, z_g_low, z_g_arch)
y_g_push = sens_y_g + push_g_y
z_g_push = sens_z_g + push_g_z

if (enableGlobal==1) & (enableSensors == 1):
    #PLOT GLOBAL SENSORS
    ax.scatter3D(sens_x_g, sens_y_g, sens_z_g, color ='blue', s=2, zorder=10)
    ax.scatter3D(sens_x_g, y_g_push, z_g_push, color =c, s=10, zorder=10)
    globalVisualize_withSensors(sens_x_g, y_g_push, z_g_push, z_g_arch, ax, c)

if enableShaker == 1:
    #PLOT THE SHAKER
    plt.scatter(6,0,marker='s',s=100,zorder=9, label='Shaker')

#DEFINE THE DAMAGE LOCATIONS
x_damage, y_damage = damageLoc()

if damageNr != [0]:
    #PLOT THE DAMAGES
    damageVisualize(damageNr,x_damage,y_damage)

plt.title(title)
plt.legend()
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
axisEqual3D(ax)
plt.tight_layout()
plt.show()
