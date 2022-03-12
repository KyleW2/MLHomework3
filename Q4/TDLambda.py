from State import State

class TDLambda:
    def __init__(self, lam: float, alpha: float, gamma: float, states: list):
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma

        # Parse state objects from list of lists
        self.states = {}
        for i in range(0, len(states)):
            actions = []

            for j in range(0, len(states[i][1])):
                actions.append(states[i][1][j])

            #print(states[i][0], actions)
            self.states[states[i][0]] = State(actions)

    def getValue(self, state: str) -> float:
        return self.states[state].getValue()
    
    def updateValue(self, state: str, reward: int, next_state: str) -> None:
        self.states[state].updateValue(self.alpha, self.gamma, reward, self.states[next_state].getValue())
    
    def updateElig(self, state: str, is_state: bool):
        self.states[state].updateElig(self.gamma, self.lam, is_state)
    
    def runEpisode(self, episode: list) -> None:
        # Episodes must go: [[state -> action -> reward], ...]
        for i in range(0, len(episode)-1):
            for k, v in self.states.items():
                # If the state equals current at time step
                if k == episode[i][0]:
                    self.updateElig(episode[i][0], True)
                    self.updateValue(episode[i][0], episode[i][2], episode[i+1][0])
                else:
                    self.updateElig(episode[i][0], False)
                    self.updateValue(episode[i][0], episode[i][2], episode[i+1][0])
            
    def printValues(self):
        for k, v in self.states.items():
            print("V(" + k + ") = " + str(v.getValue()))
            print("E(" + k + ") = " + str(v.getElig()))
            


if __name__ == "__main__":
    states = [["Olympus", ["walk", "fly"]], ["Dodoni", ["fly", "horse"]], ["Delphi", ["fly", "horse"]], ["Delos", ["fly"]]]
    test = TDLambda(0.6, 0.05, 0.9, states)

    ep1 = [["Olympus", "walk", 2], ["Dodoni", "fly", 2], ["Olympus", "fly", -1], ["Olympus"]]
    test.runEpisode(ep1)
    test.printValues()

    print()

    ep2 = [["Olympus", "fly", 2], ["Delphi", "fly", 4], ["Delos"]]
    test.runEpisode(ep2)
    test.printValues()