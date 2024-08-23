import re

def classify_equation(equation):
    # Remove spaces and handle different cases
    equation = equation.replace(" ", "")
    
    # Define regex patterns for each type of equation
    patterns = {
        "Linear": r'^[+-]?\d*x?[+-]?\d*=0$',  # Linear equation: e.g., 2x+3=0 or -x=0
        "Quadratic": r'^[+-]?\d*x\^2([+-][+-]?\d*x)?([+-][+-]?\d+)?=0$',  # Quadratic equation: e.g., x^2 - 5x + 6 = 0
        "Cubic": r'^[+-]?\d*x\^3([+-][+-]?\d*x\^2)?([+-][+-]?\d*x)?([+-][+-]?\d+)?=0$',  # Cubic equation: e.g., x^3 - 3x^2 + 3x - 1 = 0
        "Quartic": r'^[+-]?\d*x\^4([+-][+-]?\d*x\^3)?([+-][+-]?\d*x\^2)?([+-][+-]?\d*x)?([+-][+-]?\d+)?=0$',  # Quartic equation: e.g., x^4 - 4x^3 + 6x^2 - 4x + 1 = 0
        "Polynomial": r'^[+-]?\d*(x\^\d+)([+-][+-]?\d*x\^\d+)*([+-][+-]?\d+)?=0$',  # General polynomial equation
        "Exponential": r'^[+-]?\d*e\^\d+$',  # Exponential equation: e.g., 2e^x = 4
        "Logarithmic": r'^\log_\d+\(\d+\)=\d+$',  # Logarithmic equation: e.g., log_2(8) = 3
        "Trigonometric": r'^(sin|cos|tan)\(\d+\)=\d+$',  # Trigonometric equation: e.g., sin(π/2) = 1
        "Rational": r'^\d+/\d+=0$',  # Rational equation: e.g., 3/2 = 0
        "Differential": r'^d\d+/\d+=f\(\d+, \d+\)$'  # Differential equation (simplified): e.g., d^2y/dx^2 = f(x, y)
    }
    
    # Check the equation against each pattern
    for category, pattern in patterns.items():
        try:
            if re.match(pattern, equation):
                return category
        except re.error as e:
            print(f"Regex error for pattern '{pattern}': {e}")
            return "Unknown"
    
    return "Unknown"

# # Example usage
# equations = [
#     "2x + 3 = 0",
#     "x^2 - 5x + 6 = 0",
#     "x^3 - 3x^2 + 3x - 1 = 0",
#     "x^4 - 4x^3 + 6x^2 - 4x + 1 = 0",
#     "2x^5 - 3x^4 + x^3 - 7x^2 + 5x - 1 = 0",
#     "2e^x = 4",
#     "log_2(8) = 3",
#     "sin(π/2) = 1",
#     "3/2 = 0",
#     "d^2y/dx^2 = f(x, y)"
# ]

# for eq in equations:
#     print(f"Equation: {eq} -> Classification: {classify_equation(eq)}")
