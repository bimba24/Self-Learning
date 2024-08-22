import quadraticEquation

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
while True:
    thought = input("Enter your input (or 'quit' to stop): ")
    if thought.lower() == 'quit':
        break
    elif thought.lower() == 'qe':
        print("You chose to solve a quadratic equation.")
        result=input("Enter the equation:")
        final=quadraticEquation.dataMain(result)
        chain.add_thought(result)
    else:
        chain.add_thought(thought)

chain.print_chain()

