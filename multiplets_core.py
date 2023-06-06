#   Multiplets finder: scans seismic catalogs to find clusters of multiple events
#   Copyright (C) <2023>  <R. Carluccio>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>


import matplotlib.pyplot as plt
import numpy as np
import itertools 
import math
import collections

import reproducible

# create a reproducible.Context instance, that will hold all the
# tracked data.
context = reproducible.Context()

# reproducible code for recording git repository state
# here we are okay with running our code with uncommitted changes, but
# we record a diff of the changes in the tracked data.
context.add_repo(path='.', allow_dirty=True, diff=True)

# reproducible code for recording parameters; this is not necessarily needed, as the code state
# is recorded, but it is convenient.
#seed = 1
#random.seed(seed)
#context.add_data('seed', seed)

# reproducible code for exporting the provenance data to disk
context.export_yaml('results_python_prov.yaml')


# catalog file GUI requester  
#import tkinter as tk
#from tkinter import filedialog

#root = tk.Tk()
#root.withdraw()

# class for GRAPHS operations and graphics
# https://networkx.org/documentation/stable/reference/index.html
import networkx as nx

DGoptions = {
    'node_color': 'orange',
    'node_size': 400,
    'width': 1,
}

# class for operation on INTERVALS 
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

# recording parameters; this is not necessarily needed, as the code state
# is recorded, but it is convenient.
context.add_data('gardnerknopofflist', gardnerknopofflist)



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
    
# sphere to plane projection  GEO2XY
def g2x(catalogevent):
    # the last id column is't really necessary for the algorithm,
    # it can contain user reference data
    [t,lat,lon,dep,mag,id]=catalogevent
    R=6367
    alfa=180/math.pi
    x=(R*(lon - lon0)*math.cos(lat/alfa))/alfa - (R*(lon-lon0)**3*math.cos(lat/alfa)*math.sin(lat/alfa)**2)/(6.*alfa**3)
    y=((lat - lat0)*R)/alfa + ((lon - lon0)**2*R*math.sin((2.*lat)/alfa))/(4.*alfa**2)
    return [t,x,y,dep,mag,id]


# seismic catalogue import with numpy
# catfile='absolute_catalogue_file_path'
# **uncomment for GUI file requester** catfile = filedialog.askopenfilename()

# please in case data file is not edit 
catfile='Simulated_100k_005-02-2_geo.txt'

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
multiplets = []
n = 0
catalogo = 0
rj = 0

magthresh=5.5
dmplus=.4
dmminus=.6

# add to the reproducible repository the parameters used for the test
# that produces the files in ascii.zip
context.add_data('test_removed_flag', removed)
context.add_data('test_gkrad_flag', gkrad)
context.add_data('test_magthresh', magthresh)
context.add_data('test_dmplus', dmplus)
context.add_data('test_dmminus', dmminus)

########################  data subset PRE-FILTERING ######################

indexes=np.arange(len(orig))
orig=np.concatenate((orig[:,:-1], np.atleast_2d(indexes).T), axis=1)

# round necessary to overcome low precision in python default float in subtraction
evthr=round(magthresh - dmminus,6)
mp=orig[np.where(orig[:,4] >= evthr),-1][0].astype(int)
mplen=len(mp)
maxmp=mp[-1]
data=orig

# files used for intermediate computations results (not strictly necessary: used for software comprehension and debug purposes )
np.savetxt('python_orig.txt', data, fmt='%10.5f', delimiter='\t', newline='\n', header='', footer='', comments='# ', encoding=None)
np.savetxt('python_mp.txt', mp, fmt='%d')

print('magnitude subset of ', mplen, 'events...')


#*************** MAIN LOOP **************

while(len(mp)>1):
    intersections=[]
    gkconnected=[]
    gkconnectedlists=[]
    vertexlist=[]


    ######################## PIVOT point search ########################

    while(mp[n] != mp[-1] and data[mp[n], 4] < magthresh):
        n+=1

    j=mp[n]
    mpback=mp
    mp=np.delete(mp,slice(0,n))
    n=0

    ######################## POOL GKt search ########################
    connectedevents=[]
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

            # intersections (X,t,mag)
            if gkx_flag and gkt_flag and mag_flag:
                intersections.append(idp)
                vertexlist = list(set(itertools.chain.from_iterable(intersections)))

            # intersections (X,t)
            if gkx_flag and gkt_flag:
                gkconnectedlists.append(idp)
                gkconnected = list(set(itertools.chain.from_iterable(gkconnectedlists)))

            # print(idp, gkx_flag, gkt_flag, mag_flag)


        ####################  GRAPH generation for multiplets count ###################
        DG = nx.DiGraph()
        DG.add_edges_from(intersections)
        # nx.draw(DG,with_labels=True, **DGoptions)  # networkx draw()
        # plt.draw()  # pyplot draw()

        #connectedevents = []
        if j in vertexlist:
            connectedevents=list(nx.single_source_shortest_path(DG,j).keys())

    ######################## remove elements from mp and reset for a new pivot search ########################

    if len(connectedevents)>1:
        multiplets.append(connectedevents)

    lmpold=len(mp)

    if removed == "GK":
        mp=sorted(list(set(mp)-set(gkconnected)))
    elif removed == "GK-Mag":
        mp=sorted(list(set(mp)-set(intersections)))

    mp=sorted(list(set(mp)-set([j])))
    n=0

print('# multiplette', len(multiplets))

# lengths count
cnt = collections.Counter()
for mlen in [len(x) for x in multiplets]:
    cnt[mlen] += 1


print('catalogue file: ',catfile)
print('threshold magnitude:', magthresh, '(-', dmminus, ') e (+', dmplus,')' )
print('Extracted a subset of', mplen, 'events from a catalogue of ', len(orig), 'events\n')
print('algorithm parameters: removal (',removed,') interaction radii function (',gkrad,')')
print('# multiplets:', len(multiplets))
print('# counts: ',cnt)

# file containing detailed output: edit path if needed 
f=open('python_vs_out_new.txt', 'w')
print('catalogue file: ',catfile, file=f)
print('threshold magnitude:', magthresh, '(-', dmminus, ') e (+', dmplus,')' , file=f)
print('Extracted a subset of', mplen, 'events from a catalogue of ', len(orig), 'events\n', file=f)
print('algorithm parameters: removal (',removed,') interaction radii function (',gkrad,')', file=f)
print('# multiplets:', len(multiplets), file=f)
print('# counts: ',cnt, file=f)
print(multiplets, file=f)
f.close()

# reproducible: exporting the provenance data to disk
context.export_yaml('results_python_prov.yaml')