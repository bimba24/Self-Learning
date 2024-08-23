import re


def solve_quadratic(a, b, c,input_str):
    # Calculate the discriminant
    d = b**2 - 4*a*c
    print("Discriminant [2b-4ac] of this equation",input_str,"is>>",d)

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

        print("Solution 1 is >>>",x1)
        print("Solution 2 is >>>",x2)

        return f"x1 = {x1}, x2 = {x2}"

def parse_quadratic_equation(equation):
    # Remove spaces and the "= 0" part
    print("equation>>>>>>>>",equation)
    equation = equation.replace(" ", "").replace("=0", "")
    
    # Regular expression to match quadratic equations of the form ax^2 + bx + c
    pattern = r"(?P<a>-?\d*)x\^2(?P<op1>[+-]?)\s*(?P<b>\d*)x(?P<op2>[+-]?)\s*(?P<c>\d*)"
    #print("pattern>>>",pattern)
    
    # Search for the pattern in the input equation
    match = re.fullmatch(pattern, equation)
    
    if not match:
        return None
    else:
        print("The Equation is of the type ax*2+bx+c=0")
    
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
    

    print("The Coefficients and operators are:")

    print("Coefficent a in this equation is",equation,">>>",a)
    print("Coefficent b in this equation is",equation,">>>",b)
    print("Coefficent c in this equation is",equation,">>>",c)

    print("operators in this equation are ",equation,">>>",)
    print("operator1 is >>>",op1)
    print("operator2 is >>>",op2)


    return a, b, c

def dataMain(input_str):
    print("This is a Quadratic Equation , Applying , Lets Apply quadratic equation formula [x = (-b ± √ (2b - 4ac) )/2a] to find roots ")
    coefficients = parse_quadratic_equation(input_str)
    if coefficients:
        a, b, c = coefficients
        result = solve_quadratic(a, b, c,input_str)
        print("Result:", result)
    else:
        print("Invalid input format. Please try again.")
        result="Invalid Input"
    return result

# # Example usage
# input_equation = "x^2 - 5x + 6 = 0"
# dataMain(input_equation)
