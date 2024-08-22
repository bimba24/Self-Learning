# import quadraticEquation

# class Thought:
#     def __init__(self, text):
#         self.text = text
#         self.next_thought = None

# class ChainOfThoughts:
#     def __init__(self):
#         self.head = None

#     def add_thought(self, text):
#         if not self.head:
#             self.head = Thought(text)
#         else:
#             current = self.head
#             while current.next_thought:
#                 current = current.next_thought
#             current.next_thought = Thought(text)

#     def print_chain(self):
#         current = self.head
#         while current:
#             print(current.text)
#             current = current.next_thought

# # Example usage
# chain = ChainOfThoughts()
# while True:
#     thought = input("Enter the input  (or 'quit' to stop): ")
#     if thought.lower() == 'quit':
#         break
#     chain.add_thought(thought)

# chain.print_chain()

# # if(chain.contains('quadratic equation')):
# #     print("quadratic equation>>>>>",chain)

import quadraticEquation
import re

class Thought:
    def __init__(self, text):
        self.text = text
        self.next_thought = None

class ChainOfThoughts:
    def __init__(self):
        self.head = None

    def add_thought(self, text):
        if not self.head:
            self.head = Thought(text)
        else:
            current = self.head
            while current.next_thought:
                current = current.next_thought
            current.next_thought = Thought(text)

    def print_chain(self):
        current = self.head
        while current:
            print(current.text)
            current = current.next_thought
    
# Example usage
chain = ChainOfThoughts()


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

while True:
    thought = input("Enter your input (or 'quit' to stop): ")
    if thought.lower() == 'quit':
        break
    elif thought.lower() == 'qe':
        print("You chose to solve a quadratic equation.")
        # a = float(input("Enter coefficient a: "))
        # b = float(input("Enter coefficient b: "))
        # c = float(input("Enter coefficient c: "))
        # roots = quadraticEquation.solve(a, b, c)
        # result = f"The roots of the equation {a}x^2 + {b}x + {c} are: {roots}"
        result=input("Enter the equation:")
        final=quadraticEquation.dataMain(result)
        chain.add_thought(final)
        break
    else:
        # result=datatext(thought)
        # final=quadraticEquation.dataMain(thought)
        chain.add_thought(thought)

chain.print_chain()

