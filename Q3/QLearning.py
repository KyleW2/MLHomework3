from State import State

class Sarsa:
    def __init__(self, alpha: float, gamma: float, states: list):
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

    def getValue(self, state: str, action: str) -> float:
        return self.states[state].getQ(action)
    
    def updateValue(self, state: str, action: str, reward: int, next_state: str, next_action: str) -> None:
        self.states[state].updateQ(action, self.alpha, self.gamma, reward, self.states[next_state].getMaxQ())
    
    def runEpisode(self, episode: list) -> None:
        # Episodes must go: [[state -> action -> reward], ...]
        for i in range(0, len(episode)-1):
            self.updateValue(episode[i][0], episode[i][1], episode[i][2], episode[i+1][0], episode[i+1][1])
    
    def printValues(self):
        for k, v in self.states.items():
            Qs = self.states[k].getQs()

            for i in range(0, len(Qs)):
                print("Q(" + k + ", " + Qs[i][0] + ") = " + str(Qs[i][1]))
            


if __name__ == "__main__":
    states = [["Olympus", ["walk", "fly"]], ["Dodoni", ["fly", "horse"]], ["Delphi", ["fly", "horse"]], ["Delos", ["fly"]]]
    test = Sarsa(0.05, 0.9, states)

    ep1 = [["Olympus", "walk", 2], ["Dodoni", "fly", 2], ["Olympus", "fly", -1], ["Olympus", "fly"]]
    test.runEpisode(ep1)
    test.printValues()

    print()

    ep2 = [["Olympus", "fly", 2], ["Delphi", "fly", 4], ["Delos", "fly"]]
    test.runEpisode(ep2)
    test.printValues()