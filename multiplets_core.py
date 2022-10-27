import matplotlib.pyplot as plt
import numpy as np
import itertools 
import math
import collections

# catalog file GUI requester  
#import tkinter as tk
#from tkinter import filedialog

#root = tk.Tk()
#root.withdraw()

# GRAPHS
# https://networkx.org/documentation/stable/reference/index.html
import networkx as nx

DGoptions = {
    'node_color': 'orange',
    'node_size': 400,
    'width': 1,
}

# INTERVALS 
# https://pypi.org/project/portion/
import portion as P  

# useful functions definitions
def subsets(s, n):
    return list(itertools.combinations(s, n)) 

def flatten(s):
    return list(itertools.chain.from_iterable(s))
  
# Gardner Knopoff table functions for distances (km) and time (yrs)
from scipy import interpolate

gardnerknopofflist = [[2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 
     8], [19.5, 22.5, 26, 30, 35, 40, 47, 54, 61, 70, 81, 94],[6, 
     11.5, 22, 42, 83, 155, 290, 510, 790, 915, 960, 985]]

def gkr(mag):
    return np.interp(mag,gardnerknopofflist[0],gardnerknopofflist[1])

def gkt(mag):
    return np.interp(mag,gardnerknopofflist[0], [x/365 for x in gardnerknopofflist[2]])

def gkr2(m1, m2, flag):
    if flag == "sum":
        return gkr(m1)+gkr(m2)
    elif flag == "max":
        return max(gkr(m1),gkr(m2))
    elif flag=="1st":
        return gkr(m1)
    else:
        return (m1,m2,gkr(m1),gkr(m2))
    
# Caloi sphere to plane projection  GEO2XY
def g2x(listarod):
    [t,lat,lon,dep,mag,id]=listarod
    R=6367
    alfa=180/math.pi
    x=(R*(lon - lon0)*math.cos(lat/alfa))/alfa - (R*(lon-lon0)**3*math.cos(lat/alfa)*math.sin(lat/alfa)**2)/(6.*alfa**3)
    y=((lat - lat0)*R)/alfa + ((lon - lon0)**2*R*math.sin((2.*lat)/alfa))/(4.*alfa**2)
    return [t,x,y,dep,mag,id]


# seismic catalogue import with numpy
#catfile='absolute_catalogue_file_path'
# **uncomment for GUI file requester** catfile = filedialog.askopenfilename()

catfile='/Users/robertocarluccio/Downloads/Simulated_100k_005-02-2_geo.txt'

print('loading catalogue data ...')
raw=np.loadtxt(catfile)

print('records, length', raw.shape)

lat0=np.mean(raw[:,1])
lon0=np.mean(raw[:,2])

orig=np.apply_along_axis(g2x,1,raw)
print('local cartesian coordinates... ',orig.shape)
#print(orig)

# variables (RE)-INIZIALISATION

removed = "GK"
gkrad = "sum"; 
data = orig
multipletti = []
n = 0
catalogo = 0
rj = 0

magthresh=5.5
dmplus=.4
dmminus=.6


########################  data subset PRE-FILTERING ######################

indexes=np.arange(len(orig))
orig=np.concatenate((orig[:,:-1], np.atleast_2d(indexes).T), axis=1)
#data=orig[np.where(orig[:,4] >= magthresh - dmminus)]
#mp=orig[np.where(orig[:,4] >= (magthresh - dmminus)),-1][0].astype(int)

# round necessary to overcome low precision in python default float in subtraction
evthr=round(magthresh - dmminus,6)
mp=orig[np.where(orig[:,4] >= evthr),-1][0].astype(int)
mplen=len(mp)
maxmp=mp[-1]
data=orig

# put path for intermediate computations data (not strictly necessary: debug purposes)
np.savetxt('/Users/robertocarluccio/Downloads/python_orig.txt', data, fmt='%10.5f', delimiter='\t', newline='\n', header='', footer='', comments='# ', encoding=None)
np.savetxt('/Users/robertocarluccio/Downloads/python_mp.txt', mp, fmt='%d')

# indexes=np.arange(len(data))
# re-indexing on extracted sub-list
# data=np.concatenate((data[:,:-1], np.atleast_2d(indexes).T), axis=1)

print('magnitude subset of ', mplen, 'events...')


#*************** MAIN LOOP **************
#debug=open('/Volumes/GoogleDrive/Il mio Drive/Colab Notebooks/multiplette/dati/python_debug.txt', 'w')

while(len(mp)>1):
    intersezioni=[]
    gkconnected=[]
    gkconnectedlists=[]
    vertexlist=[]


    ######################## PIVOT point search ########################

    while(mp[n] != mp[-1] and data[mp[n], 4] < magthresh):
        n+=1

    j=mp[n]
    mpback=mp
    #print(j, file=debug)
    
    #if j==323387:
    #    print('breakpoint')

    mp=np.delete(mp,slice(0,n))
    n=0

    ######################## POOL GKt search ########################
    connessi=[]
    pool = [[mp[n], data[mp[n]]]]
    intpool = P.closed( data[mp[n],0], data[mp[n],0] + gkt(data[mp[n],4]))

    while(mp[n] != mp[-1]):
        nextintpool=intpool.union(P.closed(data[mp[n+1],0], data[mp[n+1],0] + gkt(data[mp[n+1],4])))

        if(mp[n] != mp[-1] and nextintpool.atomic):
            n=n+1
            intpool=nextintpool
            pool.append([mp[n],data[mp[n]]])
        else:
            break

    ######################## pool pairs generation and test over 3 criteria ########################
    if len(pool)>1:
        couples = subsets(pool, 2)


        # couples [[id1,array(t1,x1,y1,z1,mag1,idbase1)], [id2,array(t2,...)]
        for t in couples:
            idp=[t[0][0],t[1][0]]
            gkx_flag = np.linalg.norm(np.array([t[0][1][1:4]])-np.array([t[1][1][1:4]])) <= gkr2(t[0][1][4],t[1][1][4], gkrad)
            gkt_flag = round(t[1][1][0] - t[0][1][0],6) <= gkt(t[0][1][4])
            mag_flag = -dmplus <= round(data[j][4]-t[1][1][4],6) <= dmminus

            # intersezioni (X,t,mag)
            if gkx_flag and gkt_flag and mag_flag:
                intersezioni.append(idp)
                vertexlist = list(set(itertools.chain.from_iterable(intersezioni)))

            # intersezioni (X,t)
            if gkx_flag and gkt_flag:
                gkconnectedlists.append(idp)
                gkconnected = list(set(itertools.chain.from_iterable(gkconnectedlists)))

            # print(idp, gkx_flag, gkt_flag, mag_flag)


        ####################  GRAPH generation for multiplets count ###################
        DG = nx.DiGraph()
        DG.add_edges_from(intersezioni)
        # nx.draw(DG,with_labels=True, **DGoptions)  # networkx draw()
        # plt.draw()  # pyplot draw()

        #connessi = []
        if j in vertexlist:
            connessi=list(nx.single_source_shortest_path(DG,j).keys())

    ######################## remove elements from mp and reset for a new pivot search ########################

    if len(connessi)>1:
        multipletti.append(connessi)

    lmpold=len(mp)

    if removed == "GK":
        #mp1=[x for x in mp if x not in gkconnected]
        mp=sorted(list(set(mp)-set(gkconnected)))
    elif removed == "GK-Mag":
        mp=sorted(list(set(mp)-set(intersezioni)))

    mp=sorted(list(set(mp)-set([j])))
    n=0

print('# multiplette', len(multipletti))

# lengths count
cnt = collections.Counter()
for mlen in [len(x) for x in multipletti]:
    cnt[mlen] += 1


print('catalogue file: ',catfile)
print('threshold magnitude:', magthresh, '(-', dmminus, ') e (+', dmplus,')' )
print('Extracted a subset of', mplen, 'events from a catalogue of ', len(orig), 'events\n')
print('algorithm parameters: removal (',removed,') interaction radii function (',gkrad,')')
print('# multiplets:', len(multipletti))
print('# counts: ',cnt)

# put path for complete output file save
f=open('/Users/robertocarluccio/Downloads/python_vs_out_new.txt', 'w')
print('catalogue file: ',catfile, file=f)
print('threshold magnitude:', magthresh, '(-', dmminus, ') e (+', dmplus,')' , file=f)
print('Extracted a subset of', mplen, 'events from a catalogue of ', len(orig), 'events\n', file=f)
print('algorithm parameters: removal (',removed,') interaction radii function (',gkrad,')', file=f)
print('# multiplets:', len(multipletti), file=f)
print('# counts: ',cnt, file=f)
print(multipletti, file=f)
f.close()
