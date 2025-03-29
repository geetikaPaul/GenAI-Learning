class FixedSizeList:
    def __init__(self, max_length):
        self.max_length = max_length
        self.items = []

    def add(self, value):
        if len(self.items) >= self.max_length:
            self.items.pop(0)  # Remove the oldest element (first item)
        self.items.append(value)  # Add the new value

    def get_all(self):
        return self.items  # Return a copy of the list