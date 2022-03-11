
class State:
    def __init__(self, actions: list) -> None:
        self.actions = {}

        for i in range(0, len(actions)):
            self.actions[actions[i]] = 0.0
    
    def getQ(self, action) -> float:
        return self.actions[action]
    
    def updateQ(self, action, alpha, gamma, reward, next_Q) -> float:
        self.actions[action] = self.getQ(action) + (alpha * (reward + (gamma * next_Q) - self.getQ(action)))