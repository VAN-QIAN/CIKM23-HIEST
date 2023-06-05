import torch
import torch.nn.functional as F
import numpy as np

model_dict = torch.load('LA10.tar',map_location=torch.device('cpu'))
print(model_dict.keys())
dict_name = list(model_dict['model_state_dict'])
# for i, p in enumerate(dict_name):
#     print(i, p)
Mrg = F.softmax(F.relu(model_dict['model_state_dict']['acs']),dim=1).numpy()
# np.save('Mor',Mor)
print(Mrg.shape)
# for i in state_dict.children():
#     print(i)


import numpy as np
import json
# coding=utf-8

# from Tarjan import Graph as TGraph
import sys
sys.path.append("..")
from utils.CutVertex import Graph
import pandas as pd


def visualize_geo(listAss,index,df):
    # geo_file = pd.read_csv(self.geo_path, index_col=None)
    # if "coordinates" not in list(geo_file.columns):
    #     return
    geojson_obj = {'type': "FeatureCollection", 'features': []}
    # extra_feature = [_ for _ in list(geo_file.columns) if _ not in self.geo_reserved_lst]
    for row in listAss:
        # feature_dct = row[extra_feature].to_dict()
        feature_i = dict()
        feature_i['type'] = 'Feature'
        # feature_i['properties'] = feature_dct
        feature_i['geometry'] = {}
        feature_i['geometry']['type'] = 'Point'
        feature_i['geometry']['coordinates'] = eval(df['coordinates'][row])
        if len(feature_i['geometry']['coordinates']) == 0:
            return
        geojson_obj['features'].append(feature_i)

    # ensure_dir(self.save_path)
    save_name = '../data/METR_LA/geo10/'+"%s"%index + '.json'
    # print(f"visualization file saved at {save_name}")
    json.dump(geojson_obj, open(save_name, 'w',
                                encoding='utf-8'),
              ensure_ascii=False, indent=4)

original = np.load('../data/METR_LA/matrix.npy')
print(original.shape)
unique, count = np.unique(original, return_counts=True)
data_count = dict(zip(unique, count))
# print(data_count)

g1 = Graph(len(original))  # 325

for i in range(len(original)):
    for j in range(len(original)):
        if original[i][j] > 0.8: #threshhold
            g1.addEdge(i, j)
print(np.unique(original[1]))
# print("SSC in first graph ")
# g1.SCC()
# print("SSC list ")
# print(g1.res)
# # print("Dict")
# # print(g1.graph)
# print(len(g1.res))
# degree = np.zeros(len(g1.res))
# for i in range(len(g1.res)):
#     degree[i] = len(g1.res[i])
# degree = np.asarray(degree)
# print(degree)
# print((degree ** -1).shape)

print("BCC in first graph ")
g1.BCC()

# print(g1.res)
print(len((g1.res)))
# print(g1.res[1])
map=[]

for lists in g1.res:
    tmp = ()
    for sets in lists:
        # print(type(set))
        tmp += sets
    map.append(list(set(tmp)))
print(map)

indices = ()
for i in map:
    indices += tuple(i)

print("nodes reserved %d"%len(set(indices)))
# print(len(map))
degree = np.zeros(len(map))
for i in range(len(map)):
    degree[i] = len(map[i])
degree = np.asarray(degree)
# print(degree)
# print(sum(degree))
# print((degree ** -1).shape)

base = [i for i in range(len(original[0]))]
print(base)
assigned = set(indices)
notMap=set(base)-assigned
notMap = list(notMap)
print(len(notMap))
Mor = np.zeros((len(original),(len(map)+len(notMap))))
for i,nodes in enumerate(map):
    for j in range(len(nodes)):
        Mor[nodes[j]][i] = 1

for i,nodes in enumerate(notMap):
    Mor[nodes][i+len(map)] = 1


print(Mor.sum(axis=0).shape)
Mor = Mor/Mor.sum(axis=0)
print('Mor.shape is {}'.format(Mor.shape))
np.save('METR_LA.mor', Mor)
print(Mor.shape)

adj_mx_r= Mor.T @ original @ Mor
for i in range(len(adj_mx_r)):
    adj_mx_r[i][i]=1
print('new shape of adj_mx is {}'.format(adj_mx_r.shape))
# print(np.unique(adj_mx_r,return_counts=True))
np.save('../data/METR_LA.adjr',adj_mx_r)
for i in notMap:
    map.append([i])
##
# df = pd.read_csv('../data/METR_LA/METR_LA.geo')
# print(df.head())
# print(type(df['coordinates'][0]))
# print(float((df['coordinates'][0].strip('[]').split(','))[0]))
# list1 = []
# for index,i in enumerate(map):
#     # print(index,i)
#     # print(i)
#     # print(type(i))
#     length = len(i)
#     tmp = 0.
#     lan,lon = 0.,0.
#     visualize_geo(listAss=i, index=index,df=df)
#     for j in i:
#         lan += (eval((df['coordinates'][j].strip('[]').split(','))[0]))
#         lon += (eval((df['coordinates'][j].strip('[]').split(','))[1]))
#     lan, lon = lan/length,lon/length
#     coor = str([lan,lon])
#     list1.append([index,coor])

# print(list1)
# df1 = pd.DataFrame(data=list1,index=None,columns=['geo_id','coordinates'])
# df1['type'] = 'Point'
# print(df1)
# df1.to_csv("../data/METR_LA/METR_LA.geo1",index=False)

####

Mog = Mor @ Mrg
# Mog = Mog/(Mog.sum(axis=0))
MogDF = pd.DataFrame(data=Mog,index=None)
MogDF['max_idx'] = MogDF.idxmax(axis=1)
print(MogDF.head())

mapOg = []
for i in range(len(Mog[1])):
    mapOg.append((MogDF[MogDF['max_idx']==i].index.tolist()))


df = pd.read_csv('../data/METR_LA/METR_LA.geo')
print(df.head())
print(type(df['coordinates'][0]))
print(float((df['coordinates'][0].strip('[]').split(','))[0]))
list1 = []
for index,i in enumerate(mapOg):
    # print(index,i)
    # print(i)
    # print(type(i))
    length = len(i)
    tmp = 0.
    lan,lon = 0.,0.
    visualize_geo(listAss=i, index=index,df=df)
    # for j in i:
    #     lan += (eval((df['coordinates'][j].strip('[]').split(','))[0]))
    #     lon += (eval((df['coordinates'][j].strip('[]').split(','))[1]))
    # lan, lon = lan/length,lon/length
    # coor = str([lan,lon])
    # list1.append([index,coor])