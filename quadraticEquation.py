import re

def solve_quadratic(a, b, c):
    # Calculate the discriminant
    d = b**2 - 4*a*c

    # Check if the equation has real solutions
    if d < 0:
        return "No real solutions"
    elif d == 0:
        # One repeated solution
        x = -b / (2*a)
        return f"x = {x}"
    else:
        # Two distinct solutions
        x1 = (-b + d**0.5) / (2*a)
        x2 = (-b - d**0.5) / (2*a)
        return f"x1 = {x1}, x2 = {x2}"

# def parse_input(input_str):
#     # Regular expression to match quadratic equation coefficients
#     pattern = r"([+-]?\d*\.?\d+)([a-zA-Z]+)([+-]?\d*\.?\d+)([a-zA-Z]+)([+-]?\d*\.?\d+)"
#     match = re.match(pattern, input_str)

#     if match:
#         # Extract coefficients
#         a, b, c = map(float, match.groups()[::2])
#         return a, b, c
#     else:
#         # Try to parse input as a simple quadratic equation (e.g., "x^2 + 3x + 2")
#         pattern = r"([+-]?\d*\.?\d+)x\^2\s*([+-]\s*\d*\.?\d+)x\s*([+-]\s*\d*\.?\d+)"
#         match = re.match(pattern, input_str)

#         if match:
#             # Extract coefficients
#             a, b, c = map(float, match.groups())
#             return a, b, c
#         else:
#             # Try to parse input as a quadratic equation with implicit coefficients (e.g., "x^2 + 3x")
#             pattern = r"([+-]?\d*\.?\d+)x\^2\s*([+-]\s*\d*\.?\d+)x"
#             match = re.match(pattern, input_str)

#             if match:
#                 # Extract coefficients
#                 a, b = map(float, match.groups())
#                 c = 0
#                 return a, b, c
#             else:
#                 # Invalid input format
#                 return None

def parse_quadratic_equation(equation):
    # Regular expression to match quadratic equations of the form ax^2 + bx + c
    pattern = r"(?P<a>-?\d*)x\^2\s*(?P<op1>[+-]?)\s*(?P<b>\d*)x\s*(?P<op2>[+-]?)\s*(?P<c>\d*)"
    
    # Search for the pattern in the input equation
    match = re.fullmatch(pattern, equation.replace(" ", ""))
    
    if not match:
        return None
    
    # Extract coefficients and operators
    a_str = match.group('a')
    b_str = match.group('b')
    c_str = match.group('c')
    op1 = match.group('op1')
    op2 = match.group('op2')
    
    # Handle default values for coefficients
    a = int(a_str) if a_str and a_str != '-' else int(a_str + '1') if a_str == '-' else 1
    b = int(op1 + b_str) if b_str else int(op1 + '1') if op1 == '+' or op1 == '-' else 0
    c = int(op2 + c_str) if c_str else int(op2 + '1') if op2 == '+' or op2 == '-' else 0
    
    return a, b, c

def dataMain(input_str):
        coefficients = parse_quadratic_equation(input_str)
        if coefficients:
            a, b, c = coefficients
            result=solve_quadratic(a, b, c)
            print("result>>>>>>>>",result)
        else:
            print("Invalid input format. Please try again.")
        return result
    

