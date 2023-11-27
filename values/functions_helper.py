

GLOBAL_COVARIANCE_COEFFICIENTS = {}


def hash(x, y = None):
    if y is None:
        return(id(x))
    hashes = sorted([hash(x), hash(y)])
    return(hashes)


def set_cov(x, y, cov_xy):
    hashes = hash(x,y)
    
    if hashes[0] == hashes[1]:
        return(None)
    
    if hashes[0] not  in GLOBAL_COVARIANCE_COEFFICIENTS:
        GLOBAL_COVARIANCE_COEFFICIENTS[hashes[0]] = dict()
    GLOBAL_COVARIANCE_COEFFICIENTS[hashes[0]][hashes[1]] = cov_xy
    
    
def get_cov(x, y):
    hashes = hash(x,y)
    
    
    if hashes[0] == hashes[1]:
        try:
            return(x.s_v**2)
        except AttributeError:
            return(0)
    if hashes[0] not in GLOBAL_COVARIANCE_COEFFICIENTS:
        return(0)
    if hashes[1] not in GLOBAL_COVARIANCE_COEFFICIENTS[hashes[0]]:
        return(0)
    return(GLOBAL_COVARIANCE_COEFFICIENTS[hashes[0]][hashes[1]])
