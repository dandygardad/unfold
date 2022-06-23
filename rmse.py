# Root Mean Squared Error (compiler)
# # for ``unfold`` by dandy garda

import os
import json




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

dict_rmse_json_one = rmseJson_one.keys()

def compareList(l1, l2):
    l1.sort()
    l2.sort()
    if(l1 == l2):
        return true
    else:
        return false

for file in f:
    path = os.path.join(os.getcwd(), "result-rmse", file)
    
    with open(path) as file:
        rmse_json = json.load(file)

    dict_rmse_json = rmse_json.keys()

    # Check if there is same class
    is_same = compareList(dict_rmse_json_one, dict_rmse_json)

    if not is_same:
        print(f[0] + " dan " + file + " have different classes!")
        exit()


##### END OF COMPARE CLASSES START FROM FIRST JSON #####