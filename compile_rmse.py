# Root Mean Squared Error (compiler)
# # for ``unfold`` by dandy garda

import os
import json

from helper.rmse import *



##### LOAD CONFIG FROM changeData #####

f = open('changeData.json')
configJson = json.load(f)
f.close()

strictClass = configJson['rmse']['strictClass']

#### END OF LOAD CONFIG FROM changeData #####




##### LOAD FILE FROM result-rmse #####

f = os.listdir("./result-rmse")

# Make variable contain actual dist
def splitDist(n):
    return n.split('.')[0]

actual_dist = list(map(splitDist, f))

##### END OF LOAD FILE FROM result-rmse #####




##### COMPARE CLASSES START FROM FIRST JSON #####

data = dict()
path_one = os.path.join(os.getcwd(), "result-rmse", f[0])

with open(path_one) as file:
    rmse_json_one = json.load(file)

dict_rmse_json_one = list(rmse_json_one.keys())

data[actual_dist[0]] = dict()
for key in dict_rmse_json_one:
    value, _ = frequencyValue(rmse_json_one[key])
    data[actual_dist[0]] = {**data[actual_dist[0]], key: value }

for i in range(len(f)):
    if i == 0:
        continue

    path = os.path.join(os.getcwd(), "result-rmse", f[i])
    
    with open(path) as file:
        rmse_json = json.load(file)

    dict_rmse_json = list(rmse_json.keys())

    # Check if there is same class (can changed at changeData.json)
    if strictClass:
        is_same = compareList(dict_rmse_json_one, dict_rmse_json)
        if not is_same:
            errorMessage(f[0] + " and " + f[i] + " have different classes!")
            exit()

    # Append
    data[actual_dist[i]] = {}
    for key in dict_rmse_json:
        value, _ = frequencyValue(rmse_json[key])
        data[actual_dist[i]] = {**data[actual_dist[0]], key: value}

##### END OF COMPARE CLASSES START FROM FIRST JSON #####




##### MEASURE RMSE #####

result_rmse = dict()
forc_rmse = dict()

# Fetch all class
for dist in data:
    for key in data[dist]:
        result_rmse[key] = 0
        forc_rmse[key] = list()

for dist in data:
    for key in data[dist]:
        forc_rmse[key].append(data[dist][key])

for arr in forc_rmse:
    rmse = measureRMSE(forc_rmse[arr], actual_dist)
    result_rmse[arr] = rmse

# Show result
print("\n== RESULT OF RMSE (Root Mean Squared Error) ==\n``unfold`` by dandy garda\n")
for key in result_rmse:
    print(f"{key} : {result_rmse[key]}")
##### END OF MEASURE RMSE #####