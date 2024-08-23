import re
import quiMat
import solution






def extract_and_respond(text):
    # Extract the word "solve"
    match = re.search(r'\bsolve\b', text, re.IGNORECASE)
    
    if match:
        # Word "solve" is found
        command, equation = text.split(maxsplit=1)
        print("Command:", command)
        print("Equation:", equation)
        typeOfEqu=quiMat.classify_equation(equation)
        sol=solution.solution(typeOfEqu,equation)
        print("Solution>>>>:", sol)
        return "solved"
    else:
        # Word "solve" is not found
        return "Sorry Type Again"


# Example usage
input_text =input("How can I help you today!-")
response = extract_and_respond(input_text)
print(response)
