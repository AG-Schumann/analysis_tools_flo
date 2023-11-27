import numpy as np
from values.class_unit import unit
import values.objects_operations as operation






class value():
    '''
    class that takes care of the unit
    '''
    
    def __init__(self, v, s_v, units = None, fmt = ".2f", verbose = False):
        if isinstance(v, (list, tuple)):
            v = np.array(v)
        if isinstance(s_v, (list, tuple)):
            s_v = np.array(s_v)
        if isinstance(v, np.ndarray):
            self.is_array = True
            if verbose: print("v is array")
            if not isinstance(s_v, np.ndarray):
                if verbose: print("but s_v is not")
                s_v = s_v * np.ones_like(v)
            if len(s_v) != len(v):
                if verbose: print("lengths mismatch")
        else:
            self.is_array = False
        
        self.v = v
        self.s_v = s_v
        self.fmt = fmt
        
        if isinstance(units, unit):
            self.units = units
        elif isinstance(units, dict):
            self.units = unit(units)
        elif isinstance(units, str):
            self.units = unit({units: 1})
        else:
            self.units = unit(dict())
        

    # optics
    def __str__(self, format = ""):
        return(self.__format__(format = format))
    def __repr__(self, format = ""):
        return(self.__format__(format = format))
    
    def __format__(self, format = ""):
        if format == "":
                format = self.fmt
        if self.is_array is True:
            return(", ".join([
                f"{value(v, s_v, self.units):{format}}"
                for v, s_v in zip(self.v, self.s_v)
            ]))
        try:
            b_l = ""
            b_r = ""
            str_pow_10 = ""
            str_unit = ""
            pow_10 = 0
            str_unit = f" {self.units}"
            if self.s_v != 0:
                ex = int(np.log10(self.s_v))-1
                if (ex < -2) or (ex > 3):
                    pow_10 = -ex
                    str_pow_10 = f"×10^{ex}"
                str_err = f" ± {self.s_v*10**pow_10:{format}}"
                if str_unit != "" or str_pow != "":
                    b_l = "("
                    b_r = ")"
                
            else:
                str_err = ""
                ex = int(np.log10(np.abs(self.v)))
                if (ex <= -2) or (ex > 3):
                    pow_10 = -ex
                    str_pow_10 = f" × 10^{ex}"
                    format = ".1f"
                
                
            return(f"{b_l}{self.v*10**pow_10:{format}}{str_err}{b_r}{str_pow_10}{str_unit}")
        except ValueError:
            return(f"({self.v} ± {self.s_v}) {self.units}")
    
    # accesssing elements:
    def __getitem__(self, index):
        return(
            value(self.v[index], self.s_v[index], self.units, fmt = self.fmt)
        )
        
    
    
    # calculations
    def __neg__(self):
        return(-self.v, self.s_v, self.units)
    
    def __add__(self, target):
        return(operation.add(self, target))
    def __sub__(self, target):
        return(operation.sub(self, target))
    def __mul__(self, target):
        return(operation.mul(self, target))
    def __truediv__(self, target):
        return(operation.truediv(self, target))
    def __pow__(self, exp):
        return(operation.power(self, exp))
    
    # numpy stuff
    def exp(self):
        x = np.exp(self.v)
        s_x = np.abs(x * self.s_v)
        units = dict()
        return(value(x, s_x, units))
    
    
    def log(self):
        x = np.log(self.v)
        s_x = np.abs(self.s_v / self.v)
        units = dict()
        return(value(x, s_x, units))
    
    
    
    
    # all the right side stuff
    def __radd__(self, target):
        return(operation.add(self, target))
    def __rsub__(self, target):
        return(operation.sub(target, self))
    def __rmul__(self, target):
        return(operation.mul(self, target))
    def __rtruediv__(self, target):
        return(operation.rtruediv(self, target))
    def __rpow__(self, base):
        return(operation.power(base, self))
