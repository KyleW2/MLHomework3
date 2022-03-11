
class State:
    def __init__(self, actions: list) -> None:
        self.actions = {}

        for i in range(0, len(actions)):
            self.actions[actions[i]] = 0.0
    
    def getQ(self, action) -> float:
        return self.actions[action]
    
    def getQs(self) -> list:
        out = []
        for k, v in self.actions.items():
            out.append((k, v))
        return out

    
    def updateQ(self, action, alpha, gamma, reward, next_Q) -> None:
        self.actions[action] = self.getQ(action) + (alpha * (reward + (gamma * next_Q) - self.getQ(action)))