import numpy
import types
from network.gradient_tape.gradients.func_gradients import FuncGradients

def list_numpy_functions():
    """
    Lists the __name__ attribute for all callable objects in the numpy module
    that are not classes or modules. This includes NumPy universal functions (ufuncs).
    """
    function_names = []
    for name in dir(numpy):
        try:
            obj = getattr(numpy, name)
            # Check if it's callable and not a class or module
            if callable(obj) and not isinstance(obj, (type, types.ModuleType)):
                # Get the __name__ attribute
                if hasattr(obj, '__name__'):
                    function_names.append(obj.__name__)
                else:
                    # Some callable objects might not have __name__, use the attribute name
                    function_names.append(name)
        except Exception:
            # Handle potential errors when accessing attributes
            pass
    return sorted(function_names)

if __name__ == "__main__":
    numpy_functions = list_numpy_functions()
    func_gradients = dir(FuncGradients)
    print("List of __name__ for callable objects (including ufuncs) in numpy:")
    for func_name in numpy_functions:
        if func_name not in func_gradients:
            print(func_name, end=", ")
            
    print()
        
    for func_name in func_gradients:
        if func_name not in numpy_functions:
            print(func_name, end=", ")
            
    print()