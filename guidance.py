class ChainOfThought:
    def __init__(self):
        self.thoughts = []

    def add_thought(self, thought):
        """Add a new thought to the chain."""
        self.thoughts.append(thought)

    def evaluate(self):
        """Evaluate the chain of thoughts and produce a result."""
        print("Evaluating chain of thoughts...")
        result = None
        for i, thought in enumerate(self.thoughts):
            print(f"Step {i+1}: {thought}")
            # Example logic: The result is just a concatenation of thoughts
            if result is None:
                result = thought
            else:
                result += " -> " + thought
        return result

    def clear_thoughts(self):
        """Clear the chain of thoughts."""
        self.thoughts = []

# Example usage
if __name__ == "__main__":
    cot = ChainOfThought()
    
    # Adding thoughts (steps)
    cot.add_thought("Identify the problem")
    cot.add_thought("Break it down into smaller tasks")
    cot.add_thought("Solve each task step by step")
    cot.add_thought("Combine solutions to form the final answer")

    # Evaluate the chain of thoughts
    final_result = cot.evaluate()
    
    print("\nFinal Result:")
    print(final_result)
    
    # Clear thoughts if needed
    cot.clear_thoughts()
