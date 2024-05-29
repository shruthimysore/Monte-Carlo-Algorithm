import numpy as np
import matplotlib.pyplot as plt

class Model:
    episode_count = 150000
    initial_states = {"s1":0.6,"s2":0.3,"s3":0.1}
    transition_probabilities = {"s1":{"a1":{"s4":1.0},"a2":{"s4":1.0}},
                                    "s2":{"a1":{"s4":0.8,"s5":0.2},"a2":{"s4":0.6,"s5":0.4}},
                                    "s3":{"a1":{"s4":0.9,"s5":0.1},"a2":{"s4":1.0}},
                                    "s4":{"a1":{"s6":1.0},"a2":{"s6":0.3,"s7":0.7}},
                                    "s5":{"a1":{"s6":0.3,"s7":0.7},"a2":{"s7":1.0}}}
    rewards ={"s1":{"a1":7,"a2":10},
            "s2":{"a1":-3,"a2":5},
            "s3":{"a1":4,"a2":-6},
            "s4":{"a1":9,"a2":-1},
            "s5":{"a1":-8,"a2":2}
            }
    policies ={"s1":{"a1":0.5,"a2":0.5},
            "s2":{"a1":0.7,"a2":0.3},
            "s3":{"a1":0.9,"a2":0.1},
            "s4":{"a1":0.4,"a2":0.6},
            "s5":{"a1":0.2,"a2":0.8}
            }
    terminal_states = ["s6","s7"]
    gamma = 0.75
    
class MDP:
    def runEpisode(initial_state, gamma):
            state = initial_state
            discounted_reward_episode = 0
            count = 0
            while state not in Model.terminal_states:                
                states = list(Model.policies[state].keys())
                state_probabilities = list(Model.policies[state].values())
                action = np.random.choice(states,p=state_probabilities)
                available_next_states = list(Model.transition_probabilities[state][action].keys())
                next_state_probabilities = list(Model.transition_probabilities[state][action].values())
                next_state = np.random.choice(available_next_states,p=next_state_probabilities)
                reward = Model.rewards[state][action]
                state = next_state
                discounted_reward_state = (gamma ** (count)) * reward
                discounted_reward_episode += discounted_reward_state
                count += 1
            return discounted_reward_episode

def main():
    rewards_list = []
    J_cap = []    
    total_cumulative_reward = 0
    for i in range(Model.episode_count):
        initial_state = np.random.choice(list(Model.initial_states.keys()),p=list(Model.initial_states.values()))
        discounted_reward_episode = MDP.runEpisode(initial_state,Model.gamma)
        rewards_list.append(discounted_reward_episode)
        total_cumulative_reward += discounted_reward_episode
        J_cap.append(total_cumulative_reward/(i+1))
    average_discounted_reward = np.mean(rewards_list)
    variance_discounted_reward = np.var(rewards_list)
    print(f"Average of discounted rewards = {average_discounted_reward}")
    print(f"Variance of discounted rewards = {variance_discounted_reward}")
    x = list(range(1,Model.episode_count+1))
    y = J_cap
    plt.xlabel("Number of episodes")
    plt.ylabel("Estimated J(π)")
    plt.title("Estimated J(π) vs Number of episodes")
    plt.plot(x,y)    
    plt.show()

if __name__ == "__main__":
    main()