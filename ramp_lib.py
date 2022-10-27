#!/software/anaconda3/envs/server/bin/python
from Doberman import dispatcher, Database
from pymongo import MongoClient

import os
from datetime import datetime, timedelta
import time
import numpy as np
import requests
import json

log_file = "cryocon.log"



mongo_URI = 'mongodb://webmonitor:42RKBu2QyeOUHkxOdHAhjfIpw1cgIQVgViO4U4nPr0s=@192.168.131.2:27017/admin'
client = MongoClient(mongo_URI)
db = Database(client, experiment_name='xebra')


def update_log(text, file = log_file, mode = "a", end = "\n"):
    print(text, end = end)
    with open(file, mode) as f:
        f.write(f"{text}{end}")



def set_target(temp):
    command = f"cryocon_22c setpoint 1 {temp:.3f}"
    print(f"{datetime.now()} Sending command: {command}")
    dispatcher.process_command(db, command)

def get_target():
    url = "http://192.168.131.2:8086/query?db=xebra"

    query_string = f'SELECT last("setpoint1") FROM "temperature"'
    result = json.loads(requests.post(url,
        data = {"q":query_string},
    ).text)

    
    return(result["results"][0]["series"][0]["values"][0][1])
    
    
def T_profile(dt, dT, T0 = 0, k = 8):
    '''
    duration: duration of temperature profile in seconds
    d_temp: change in temperature in K
    k: width of slope (higher means wider)
    '''
    
    # calculate values
    ts = np.arange(0, dt, .1)
    dts = 1 / (1 + np.exp(-(ts-(dt/2))/dt*k))
    
    # propper scaling from T0 + dT*(0 to 1)
    Ts = T0 + dT * (dts - dts[0])/np.abs(dts[0]-dts[-1])
    
    
    
    Ts = np.round(Ts, 3)
    
    return (ts, Ts)