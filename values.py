import numpy as np
import warnings

class unit():
    '''
    a class that is able to take care of all the unit stuff
    '''
    def __init__(self, units = None):
        if isinstance(units, dict):
            self.units = {u:e for u, e in units.items() if ((e != 0) and (u != ""))}
        elif isinstance(units, str):
            self.units = {units: 1}
        else:
            self.units = dict()
        self.str_unit = " ".join([self._nice_unit(u, e) for u, e in self.units.items()])

    

    def _nice_unit(self, u, e):
        if e == 0:
            return("")
        if e == 1:
            return(f"{u}")
        return(f"{u}^{e}")
        
        
    
    def __repr__(self):
        return(" ".join([self._nice_unit(u, e) for u, e in self.units.items()]))
    # def __str__(" ".join([self._nice_unit(u, e) for u, e in self.units.items()])):
        # return(self.__repr__())
    # def __format__(self, format=""):
        # return(self.__repr__())
    
    def __len__(self):
        return(len(self.units))
    
    def __bool__(self):
        return(len(self.units) > 0)
    
    def __eq__(self, target):
        return(self.units == target.units)
    
    def __add__(self, target):
        if self != target:
            w_string = f"units mismatch in addition!: {self.units} != {target.units}"
            warnings.warn(w_string)
        return(self.units)
    
    def __sub__(self, target):
        if self != target:
            w_string = f"units mismatch in subtraction!: {self.units} != {target.units}"
            warnings.warn(w_string)
        return(self.units)

    def __mul__(self, target):
        new_units = {**self.units}
        for u, e in target.units.items():
            if u in new_units:
                new_units[u] += e
            else:
                new_units[u] = e
        return(new_units)
    
    def __truediv__(self, target):
        new_units = {**self.units}
        for u, e in target.units.items():
            if u in new_units:
                new_units[u] -= e
            else:
                new_units[u] = -e
        return(new_units)
    
    def __pow__(self, exp):
        new_units = {u:e*exp for u, e in self.units.items()}
        return(new_units)
    
    
    def reziproc(self):
        new_units = {u:-e for u, e in self.units.items()}
        return(new_units)


class value():
    '''
    class that takes care of the unit
    '''
    def __init__(self, v, s_v, units = None):
        self.v = v
        self.s_v = s_v
        
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
            format = ".1f"
        
        b_l = ""
        b_r = ""
        str_pow_10 = ""
        str_unit = ""
        pow_10 = 0
        ex = int(np.log10(self.s_v))
        if (ex <= -2) or (ex > 3):
            pow_10 = -ex
            str_pow_10 = f" × 10^{ex}"
            format = ".1f"

        if bool(self.units) is True or str_pow != "":
            b_l = "("
            b_r = ")"
            str_unit = f" {self.units}"
            
        return(f"{b_l}{self.v*10**pow_10:{format}} ± {self.s_v*10**pow_10:{format}}{b_r}{str_pow_10}{str_unit}")
    
    # calculations
    def __add__(self, target):
        if isinstance(target, value):
            x = self.v + target.v
            s_x = (self.s_v**2 + target.s_v**2)**.5
            units = self.units + target.units
        else:
            x = self.v + target
            s_x = self.s_v 
            units = self.units
        return(value(x, s_x, units))
    
    def __sub__(self, target):
        if isinstance(target, value):
            x = self.v - target.v
            s_x = (self.s_v**2 + target.s_v**2)**.5
            units = self.units + target.units
        else:
            x = self.v - target
            s_x = self.s_v
            units = self.units
        return(value(x, s_x, units))
        
    def __rmul__(self, target):
        if isinstance(target, value):
            x = self.v * target.v
            s_x = x * ((self.s_v/self.v)**2 + (target.s_v/target.v)**2)**.5
            units = self.units * target.units
        else:
            x = self.v * target
            s_x = self.s_v * target
            units = self.units
        return(value(x, s_x, units))
    
    def __mul__(self, target):
        if isinstance(target, value):
            x = self.v * target.v
            s_x = x * ((self.s_v/self.v)**2 + (target.s_v/target.v)**2)**.5
            units = self.units * target.units
        else:
            x = self.v * target
            s_x = self.s_v * target
            units = self.units
        return(value(x, s_x, units))
    
    def __truediv__(self, target):
        if isinstance(target, value):
            x = self.v / target.v
            s_x = x * ((self.s_v/self.v)**2 + (target.s_v/target.v)**2)**.5
            units = self.units * target.units
        else:
            x = self.v / target
            s_x = self.s_v / target
            units = self.units
        return(value(x, s_x, units))
    def __rtruediv__(self, target):
        if isinstance(target, value):
            x = target.v / self.v
            s_x = x * ((self.s_v/self.v)**2 + (target.s_v/target.v)**2)**.5
            units = target.units / self.units
        else:
            x = target / self.v
            s_x = 1/self.v**2 * self.s_v
            units = self.units.reziproc()
        return(value(x, s_x, units))
    
    def __pow__(self, exp):
        if isinstance(exp, value):
            raise TypeError("value ** value not yet implemented")
        else:
            x = self.v ** exp
            s_x = exp * self.v**(exp-1) * self.s_v
            units = self.units**exp
        return(value(x, s_x, units))
