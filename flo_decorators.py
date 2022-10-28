import sys

def silencer(f):
    '''
    decorator that adds argument "silent" to function:
    if set to True: sets the stdout to dev/null during call    
    '''

    def wrapper(*args, silent = False, **kwargs):
        ret = None
        sys_stdout = sys.stdout
        if silent is True: 
            sys.stdout = open('/dev/null', 'w')
        try:
            ret = f(*args, **kwargs)
        finally:
            # just make sure the stdout is reset
            pass
            sys.stdout = sys_stdout
            return(ret)
        
    return wrapper