import traceback
from datetime import datetime
import time
import numpy as np
from analysis_help import clear

def N_dim_grid_creator(lists):
    '''
    creates all permutations of lists
    '''
    names = list(lists)
    arrays = [l for l in lists.values()]
    grid_ = np.array(np.meshgrid(*arrays)).T.reshape(-1, len(lists))
    grid = [0]*len(grid_)
    for i, values in enumerate(grid_):
        grid[i] = {n:v  for n,v in zip(names, values)}
    
    return(grid)




def list_iterator(f, grid, header = None, __KeyboardInterrupt = True, clear_std = True, *args, **kwargs):
    '''
this iterator wil iterate over call f with all enties of grid

=== Inputs ===
- f
f(A, B, *args, **kwargs) is a function that accepts one entry of list and
  *args + **kwargs 

- grid
[
    {
        "A": 1,
        "B": "foo", 
    },
    {
        "A": 2,
        "B": "bar", 
    },
    
    {...}
]
header: a string that will be printed every Iteration
__KeyboardInterrupt: wheter or not this function catches keyboard interupts
  set to anything but True to tprevent this fucntion to catch Keyboard interrupts
*args and **kwargs that will be passed down to f

=== Returns five values === 
(row_i, row, pars, True, result)
  - the permutation i
  - the values of row_i as list
  - the paramaters as dict
  - whether the function call was sucessfull
  - the result of the function call

=== USE === 
for i, row, par, succsess, res in N_dim_iterator(f, lists):
    print(f'{i} {par["run"]}: {res} ')

      
    
    '''
    
    if __KeyboardInterrupt is True:
        __KeyboardInterrupt = KeyboardInterrupt
        
    else:
        __KeyboardInterrupt = ()
    
    
    try:
        # initialisation
        global failed
        failed = []
        sucess = []


        len_total = len(grid)
        str_len_total = len(str(len_total)) 
        

        t0 = datetime.now()


        for row_i, row in enumerate(grid):
            ret = {
                "i": row_i,
                "pars": None,
                "result": None, 
                "error": (None, None),
            }
            
            t1 = datetime.now()
            
            
            # calc ETA
            str_out = f'{row_i:>{str_len_total}}/{len_total}'
            
            if row_i > 0:
                dur = t1 - t0
                eta = dur / (row_i) * (len_total - row_i - 1)
                str_out += f" ETA: {eta}"
            
            # check failed:
            if len(failed) > 0:
                str_out += f" \33[0mfailed: {len(failed)}\33[0m"
            if clear_std:
                clear()
                
            print(f"{str_out}")
            if header is not None:
                print(header)
            
            
            try:
                
                pars = {n:v for n,v in row.items()}
                ret["pars"] = pars
                
                for name, value in pars.items():
                    print(f"  \33[34m{name}:\33[0m {value}")
            
                
                result = f(*args, **pars, **kwargs)
                ret["result"] = result
                sucess.append(row_i)
            
            except Exception as e:
                ret["error"] = (e, traceback.format_exc())
                failed.append(ret)
                
            
            yield(ret)
            
            
            
    except __KeyboardInterrupt:
        failed.append((-1, {}, [], "Stopped by hand", "Stopped by hand"))
    if clear_std:
        clear()
    
    print(f"\n\33[32mDone with {len(sucess)}/{len_total} iterations")
    if len(failed) > 0:
        print(f"\n\33[31mfailed ({len(failed)}):\33[0m")
        for fail in failed:
            print(f'\n\33[34m{fail["i"]}: \33[31m{fail["error"][0]}\33[0m')
            print(f'\33[35m{fail["pars"]}\33[0m')
            print(fail["error"][1])
            
            
    