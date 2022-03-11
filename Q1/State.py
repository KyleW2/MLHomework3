
class State:
    def __init__(self, actions: list, probabilities = None, value = None) -> None:
        self.actions = {}

        for i in range(0, len(actions)):
            if probabilities != None:
                self.actions[actions[i]] = probabilities[i]
            else:
                self.actions[actions[i]] = 0.0
        
        self.value = 0.0
        if value != None:
            self.value = value
    
    def getValue(self) -> float:
        return self.value
    
    def updateValue(self, alpha, gamma, reward, next_value) -> float:
        self.value = self.value + (alpha * (reward + (gamma * next_value) - self.value))