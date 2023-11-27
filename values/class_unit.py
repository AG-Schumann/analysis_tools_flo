import warnings
warnings.filterwarnings("always")


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
    
    def __rmul__(self, target):
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
