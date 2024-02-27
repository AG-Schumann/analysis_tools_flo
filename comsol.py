import numpy as np
import pandas as pd


def get_dict_from_str(str_, delim1 = "; ",delim2 = "=", prefix = ""):
    return(
        {f"{prefix}{k}":v for k, v in [x.split(delim2) for x in str_.split(delim1)]}
    )
    

def _clean_one_name(name):
    name = name.split(" ")[0]
    if "." in name:
        name = name.split(".")[1]
    return(name)

def _clean_names(df):
    names = {n:_clean_one_name(n) for n in df.columns}
    df = df.rename(columns = names)
    return(df)
    
    

def read_csv(path, unpack_settings = True, explicit_settings = True, clean_names = True):
    '''
    
    returns as nice pandas dataframe with comsol results
    
    path:
        path to comsol .csv (!!!) file
    
    unpack_settings (True):
        if parameter sweep is done comsol adds extra colums to file which is anoying
        if this is true, new rows will be created per parameter set and the names unified
        also a column with the parameterset name is created
    
    explicit_settings (True):
        unpack the setting such that each is available as a separate row
        no conversion to float / int or so is performed!
    
    clean names (True):
        removes units and such from column names
        
        
    '''
    
    
    
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if line[0] == "%":
                names = line[2:-1].replace(", ", "; ").split(",")
            else:
                break

    df = pd.read_csv(
        path,
        comment = "%",
        header = None,
        names = names
    )
    if ("@" not in "".join(names)) or (unpack_settings is False):
        if clean_names is True:
            df = _clean_names(df)
        return(df)
    sets = dict()
    names_keep = []

    for name in names:
        if "@" in name:
            set_ = name[name.index("@")+2:]
            if set_ in sets:
                sets[set_].add(name)
            else:
                sets[set_] = {name}
        else:
            names_keep.append(name)

    df2 = pd.DataFrame()
    for set_name, set_names in sets.items():
        df_s = df[[*names_keep, *set_names]]

        new_names = {name:name[:name.index("@")-1] for name in set_names}
        df_s = df_s.rename(columns = new_names)

        df_s["setting"] = set_name
        df2 = pd.concat((df2, df_s), sort=False)
    
    if clean_names is True:
        df2 = _clean_names(df2)
    
    if explicit_settings is True:
        sdf = pd.DataFrame([get_dict_from_str(setting_, prefix = "setting_") for setting_ in df2.setting])
        
        # merging does not work! all values are the same, wtf....
        for c in sdf.columns:
            df2[c] = sdf[c].values
    
    
    return(df2)
