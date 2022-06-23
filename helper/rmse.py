# RMSE (Root Mean Squared Error)
# for ``unfold`` by dandy garda

import os
import json

def saveData(name, data):
    if not os.path.exists(os.getcwd() + '\\result-rmse'):
        os.makedirs('result-rmse')
    with open("./result-rmse/"+ str(name) +".json", 'w+') as result:
        json.dump(data, result)
        print("\nSaved in " + './result-rmse/' + str(name) + '.json', end="\n")