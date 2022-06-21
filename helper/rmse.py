# RMSE (Root Mean Squared Error)
# for ``unfold`` by dandy garda

import os
import json

def saveData(name, data):
    if not os.path.exists(os.getcwd() + '\\rmse'):
        os.makedirs('rmse')
    with open("./rmse/"+ str(name) +".json", 'w+') as result:
        json.dump(data, result)
        print("\nSaved in " + './rmse/' + str(name) + '.json', end="\n")