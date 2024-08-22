# #input="x^2 + 3x + 2"

# import re

# def parse_quadratic_equation(equation):
#     # Regular expression to match quadratic equations of the form ax^2 + bx + c
#     pattern = r"(?P<a>-?\d*)x\^2\s*(?P<op1>[+-]?)\s*(?P<b>\d*)x\s*(?P<op2>[+-]?)\s*(?P<c>\d*)"
    
#     # Search for the pattern in the input equation
#     match = re.fullmatch(pattern, equation.replace(" ", ""))
    
#     if not match:
#         return None
    
#     # Extract coefficients and operators
#     a_str = match.group('a')
#     b_str = match.group('b')
#     c_str = match.group('c')
#     op1 = match.group('op1')
#     op2 = match.group('op2')
    
#     # Handle default values for coefficients
#     a = int(a_str) if a_str and a_str != '-' else int(a_str + '1') if a_str == '-' else 1
#     b = int(op1 + b_str) if b_str else int(op1 + '1') if op1 == '+' or op1 == '-' else 0
#     c = int(op2 + c_str) if c_str else int(op2 + '1') if op2 == '+' or op2 == '-' else 0
    
#     return a, b, c

# def main():
#     equation = input("Enter a quadratic equation (e.g., x^2 + 3x + 2): ")
#     result = parse_quadratic_equation(equation)
    
#     if result:
#         a, b, c = result
#         print(f"The coefficients are:\na = {a}\nb = {b}\nc = {c}")
#     else:
#         print("The input is not a valid quadratic equation.")

# if __name__ == "__main__":
#     main()


import re

def datatext(inp):
    equation=inp
    # Remove spaces and ensure the equation ends with =0
    equation = equation.replace(" ", "")
    if not equation.endswith('=0'):
        return None

    # Regular expression to match quadratic equations of the form ax^2+bx+c=0
    pattern = r"^(?P<a>-?\d*)x\^2(?P<op1>[+-])(?P<b>\d*)x(?P<op2>[+-])(?P<c>\d*)=0$"
    
    # Search for the pattern in the input equation
    match = re.fullmatch(pattern, equation)
    
    if not match:
        return None
    else:
        return equation
    
    # # Extract coefficients and operators
    # a_str = match.group('a')
    # b_str = match.group('b')
    # c_str = match.group('c')
    # op1 = match.group('op1')
    # op2 = match.group('op2')
    
    # # Handle default values for coefficients
    # a = int(a_str) if a_str and a_str != '-' else int(a_str + '1') if a_str == '-' else 1
    # b = int(op1 + b_str) if b_str else int(op1 + '1') if op1 == '+' or op1 == '-' else 0
    # c = int(op2 + c_str) if c_str else int(op2 + '1') if op2 == '+' or op2 == '-' else 0
    
    # return a, b, c

def main():
    equation = input("Enter a quadratic equation (e.g., x^2+3x+2=0): ")
    result = datatext(equation)
    
    if equation:
        # a, b, c = result
        print(f"The equation is>>>>",equation)
    else:
        print("The input is not a valid quadratic equation.")

if __name__ == "__main__":
    main()

