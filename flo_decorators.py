import sys

def silencer(f):
    '''
    decorator that adds argument "silent" to function that temporarily sets the stdout to dev/null
    
    '''

    def wrapper(*args, silent = False, **kwargs):
        
        sys_stdout = sys.stdout
        if silent is True: 
            sys.stdout = open('/dev/null', 'w')
        
        ret = f(*args, **kwargs)
        sys.stdout = sys_stdout
        return(ret)
        
    return wrapper