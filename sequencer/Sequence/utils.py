

from sympy import symbols
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify
import numpy as np

def tuple_to_string(tuple_input):
    """
    Convert a tuple of three strings to one string.
    
    Args:
    tuple_input (tuple): A tuple containing three strings.
    
    Returns:
    str: A single string combining the three input strings.
    """
    if len(tuple_input) != 3:
        raise ValueError("Input tuple must contain exactly three strings.")
    
    return "|".join(tuple_input)

def string_to_tuple(string_input):
    """
    Convert one string to a tuple of three strings.
    
    Args:
    string_input (str): A string containing three parts separated by '|'.
    
    Returns:
    tuple: A tuple containing three strings.
    """
    parts = string_input.split("|")
    
    if len(parts) != 3:
        raise ValueError("Input string must contain exactly two '|' separators.")
    
    return tuple(parts)



def exp_to_func(exp_str):
    x = symbols('t')
    expr = parse_expr(exp_str)
    return lambdify(x, expr)
def find_min_max(func, start, end, step=0.000001):
    """
    Find the minimum and maximum values of a function within a given range using NumPy.
    
    Args:
    func (callable): The function to evaluate
    start (float): The start of the range
    end (float): The end of the range
    step (float): The step size for the range (default is 1)
    
    Returns:
    tuple: (min_value, max_value, t_min, t_max)
    """
    t_range = np.arange(start, end + step, step)
    values = func(t_range)
    
    max_value = np.max(values)
    min_value = np.min(values)
    
    t_max = t_range[np.argmax(values)]
    t_min = t_range[np.argmin(values)]
    
    return min_value, max_value, t_min, t_max


