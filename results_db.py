import numpy as np
from  analysis_help import clear
from mystrax import *
import pickle
import shutil
import pandas as pd


# define variables
database_file = "/data/workspace/Flo/library_flo/results_db/results.pkl"
voltage_file = "/data/workspace/Flo/library_flo/Voltages_lookup.xls"


voltages_LUT = pd.read_excel(voltage_file)

def get_voltages(run, context = "s", voltages_LUT = voltages_LUT):
    result = voltages_LUT.loc[
          (voltages_LUT["run"] == run)
        & (voltages_LUT["context"] == context)
    ].to_dict(orient = "records")[0]
    del result["run"]
    del result["context"]

    return(result)

def find_ids_in_db(database, run, context, config):
    return([
        i
         for i, x in enumerate(database)
         if (
                 (x["meta"]["run"] == run)
             and (x["meta"]["context"] == context)
             and (x["meta"]["config"] == config)
        )
    ])
    
    



def get_analysis_groups_from_db(database, analysis):
    groups_names = {x["meta"]["groups"][analysis] for x in database if analysis in x["meta"]["groups"]}
    groups = {
        group:[
            x for x in database if x["meta"]["groups"][analysis] == group
        ]
        for group in groups_names
    }

    return(groups)



def create_template_from_run(
    run,
    remark = "",
    context = "s",
    config = {},
    groups = {},
    V_Cathode = "auto",
    V_Gate = "auto",
    V_Anode = "auto",
    E_drift = "auto",
    E_extract = "auto",
):
    out = {
        "meta":{},
        "settings":{},
        "results":{},
    }
    
    
    
    run_str = f"{run:0>5}"
    
    rundocs = list(db.runs.find({"experiment": experiments[context], "run_id": run}))
    if len(rundocs) != 1:
        raise KeyError(f'did not find exactly one result on database for {{"experiment": {experiments[context]}, "run_id": {run}}}')
    else:
        rundoc = rundocs[0]
        
    out["meta"]["run"] = run
    out["meta"]["run_str"] = run_str
    out["meta"]["remark"] = remark
    out["meta"]["context"] = context
    out["meta"]["config"] = config
    out["meta"]["start"] = rundoc["start"]
    out["meta"]["duration"] = (rundoc["end"] - rundoc["start"]).total_seconds()
    out["meta"]["groups"] = groups
    
    
    try:
        voltages_tab = get_voltages(run, context, voltages_LUT)
    except:
        voltages_tab = False
    
    if voltages_tab is not False:
        out["settings"] = voltages_tab    
    else:
        try:
            regex_results = fhelp.regex_comment(rundoc["comment"])
        except:
            regex_results = {}
        
        V_auto_failed = []
        if V_Cathode == "auto":
            if "cathode_mean" in rundoc:
                V_Cathode = -abs(rundoc["cathode_mean"]/1000)
            elif("cathode" in regex_results):
                V_Cathode = regex_results["cathode"]
            else:
                V_Cathode = -5.500
                V_auto_failed.append("V_Cathode")
        
        if V_Gate == "auto":
            if "gate_mean" in rundoc:
                V_Gate = -abs(rundoc["gate_mean"]/1000)
            elif("screen" in regex_results):
                V_Gate = regex_results["screen"]
            else:
                V_Gate = -2.000
                V_auto_failed.append("V_Gate")
        
        if V_Anode == "auto":
            if "anode_mean" in rundoc:
                V_Anode = abs(rundoc["anode_mean"]/1000)
            elif("anode" in regex_results):
                V_Anode = regex_results["anode"]
            else:
                V_Anode = 2.500
                V_auto_failed.append("V_Anode")
        
        
        

        
        out["settings"]["V_Cathode"] = V_Cathode
        out["settings"]["V_Gate"] = V_Gate
        out["settings"]["V_Anode"] = V_Anode
        
        out["settings"]["dV_Cathode"] = V_Gate-V_Cathode
        out["settings"]["dV_Anode"] = V_Anode-V_Gate
    
        
        if E_drift == "auto":
            out["settings"]["E_drift"] = out["settings"]["dV_Cathode"]/7
        else:
            out["settings"]["E_drift"] = E_drift
        if E_extract == "auto":
            out["settings"]["E_extract"] = out["settings"]["dV_Anode"]/.5
        else:
            out["settings"]["E_extract"] = E_extract
        if len(V_auto_failed) > 0:
            out["settings"]["V_auto_failed"] = V_auto_failed
        
        
    
    return(out)





def load_database(database_file = database_file, context = False):
    try:
        with open(database_file, "rb") as f:
            database = pickle.load(f)
        if context is not False:
            database = [dbe for dbe in database if dbe["meta"]["context"] == context]

        print(f"\33[32mloaded Database into \33[1mdatabase\33[0m (n={len(database)})")
    except Exception as e:
        print(f"\33[31m{e}\33[0m")
        database = []
        print("created empty database file locally")
    
    
    return(database)
    
def write_database(database, database_file = database_file, backup = True, verbose = False):
    try:
        if backup is True:
            shutil.copy(database_file, f"{database_file}.backup.{datetime.now()}")
        with open(database_file, "wb") as f:
            pickle.dump(database, f)
        if verbose is True:
            print(f"\33[32mwrote Database into \33[1m{database_file}\33[0m")
    except Exception as e:
        print(f"\33[31m{e}\33[0m")
    return(None)



