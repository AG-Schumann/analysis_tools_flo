import re
from datetime import datetime



try:
    from IPython.display import clear_output
    def clear():
        clear_output(True)
except:
    def clear():
        pass




def regex_comment(comment):
    m = re.search('ADC.*= ([0-9]+); Anode ([+-]?\d+\.?\d?) kV, Cathode ([+-]?\d+\.?\d?) kV, Screen ([+-]?\d+\.?\d?) kV', comment)
    return(
        {
            "adc": int(m.group(1)),
            "anode": float(m.group(2)),
            "cathode": float(m.group(3)),
            "screen": float(m.group(4)),
        }
    )
    
    
def get_ETA(t_start, i, N):
    if i > 0:
        return((datetime.now() - t_start) / i * (N-i))
    else:
        return(float("inf"))
