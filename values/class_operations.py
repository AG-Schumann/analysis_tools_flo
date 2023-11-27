import numpy as np
import values.functions_helper

class operation():
    def __init__(self, function, derivatives, unit_operation = False):
        
        # adjusting types
        if callable(derivatives) or len(derivatives) == 1:
            derivatives = [derivatives, lambda *args: 1]
        
        
        self.function = function
        self.derivatives = derivatives
        self.unit_operation = unit_operation
        
    def __call__(self, x, y = None):
        
        u1 = values.class_unit.unit({})
        if isinstance(x, values.value):
            v1  = x.v
            s_v1 = x.s_v
            u1 = x.units
        elif isinstance(x, (int, float)):
            v1 = x
            s_v1 = 0
        elif isinstance(x, (np.ndarray)):
            v1 = x.v
            s_v1 = np.zeros_like(v1)
        else:
            v1 = 0
            s_v1 = 0
        
            
        
        u2 = values.class_unit.unit({})
        if isinstance(y, values.value):
            v2 = y.v
            s_v2 = y.s_v
            u2 = y.units
        elif isinstance(y, (int, float)):
            v2 = y
            s_v2 = 0
        elif isinstance(y, (np.ndarray)):
            v2 = y
            s_v2 = np.zeros_like(v2)
        else:
            v2 = 0
            s_v2 = 0
            
        

        args = [v1, v2]
        s_args = [s_v1, s_v2]
        cov = values.functions_helper.get_cov(x, y)

        
        res = self.function(*args)
        var_res = (self.derivatives[0](*args) * s_v1)**2 \
                + (self.derivatives[1](*args) * s_v2)**2 \
                + 2 * self.derivatives[0](*args) * self.derivatives[1](*args) * cov
                
        s_res = var_res **.5
        if  self.unit_operation is False:
            units = self.function(u1, u2)
        else:
            units = self.unit_operation(v1, v2, s_v1, s_v2, u1, u2)

        
        return(values.value(res, s_res, units))
        
        
        
        
    
