import collections
from random_environment import Environment
import numpy as np
import torch


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length (you will need to increase this)
        self.episode_length = 250
        # when episode ends reset
        self.num_steps_taken = 0
        # episodes_count count
        self.episodes_count = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # Store the initial state of the agent
        self.init_state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        self.dqn = DQN()
        self.action_space = np.array([0, 1, 2, 3])
        self.step_size = 0.02
        self.debug_optimal_policy = True
        '''
        #original actions:
        self.continuous_actions = self.step_size * np.array([
            [1, 0],# right
            [0, 1],# up
            [-1, 0],# left
            [0, -1]# down
        ], dtype=np.float32)
        '''
        # modified actions:
        self.continuous_actions = self.step_size * np.array([
            [0, 1],# up
            [1, 0],# right
            [0, -1],# down
            [-1, 0]# left
        ], dtype=np.float32)

        self.epsilon = 1
        self.delta = 0.000015
        # evaluate policy every n step
        self.evaluate_yes_or_no = False
        # save functioning greedy pi
        self.best_case = save_best_weights()
        self.is_it_pi_star_now = False
        self.optimal_policy_loaded = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.episodes_count += 1
            self.evaluate_yes_or_no = False

            self.dqn.q_network.train()
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):

        # Periodically evaluate the policy
        if self._should_evaluate_policy():
            self.evaluate_yes_or_no = True
            self.dqn.q_network.eval()

        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1

        # Evaluate the policy
        if self.evaluate_yes_or_no:
            return self.get_greedy_action_for_evaluation(state)

        # Decrease the episode length so that it converges to 100
        if self.num_steps_taken > 10000 and self.episodes_count % 10 == 0 and self.num_steps_taken % self.episode_length == 1:
            self.decrease_episode_length(delta=25)

        # Store the state; this will be used later, when storing the transition
        self.state = state

        # Make action
        action = self.e_greedy_action()

        # Get the continuous action
        continuous_action = self._discrete_action_to_continuous(action)
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Don't train the network when evaluating it
        if self.evaluate_yes_or_no:

            if self.best_case.min_steps_to_goal == 1000:
                self.best_case.weights_without_goal_but_min_dist(
                    distance=distance_to_goal,
                    weights=self.dqn.q_network.state_dict()
                )

            # otherwise consider only the solutions which reach the goal
            if distance_to_goal < 0.03:
                steps = self.num_steps_taken % self.episode_length
                steps_taken = steps if steps != 0 else self.episode_length
                steps_taken -= 1
                if steps_taken < self.best_case.get_min_steps_to_goal():
                    weights = self.dqn.q_network.state_dict()
                    self.best_case.save_weights(
                        num_steps=steps_taken,
                        weights=weights
                    )
                    if self.debug_optimal_policy:
                        print('Greedy policy works! Reached goal in {} steps.'.format(
                            steps_taken
                        ))
            return

        # Convert the distance to a reward
        reward = self.calculate_reward(next_state, distance_to_goal)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        # Add transition to the buffer
        self.dqn.replay_buffer.add(transition)
        if self.dqn.replay_buffer.is_big_enough():
            self.dqn.train_q_network()
            self.epsilon = max(self.epsilon-self.delta, 0.15)

        # Update target network every 50th step
        if self.num_steps_taken % 50 == 0:
            self.dqn.update_target_network()

    def calculate_reward(self, next_state, distance_to_goal):
        reward = 1 - distance_to_goal
        if distance_to_goal <= 0.2:
            reward *= 3
        elif distance_to_goal <= 0.3:
            reward *= 2
        elif distance_to_goal <= 0.5:
            reward *= 1.5

        if not np.any(self.state - next_state):
            reward /= 1.5
        return reward

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        self._load_best_policy_and_weights()

        action_rewards = self.dqn.q_network.forward(torch.tensor(state)).detach().numpy()
        discrete_action = np.argmax(action_rewards)
        return self._discrete_action_to_continuous(discrete_action)

    # Function to get the greedy action for a particular state when evaluating stuff
    def get_greedy_action_for_evaluation(self, state):
        action_rewards = self.dqn.q_network.forward(
            torch.tensor(state)
        ).detach().numpy()
        discrete_action = np.argmax(action_rewards)
        return self._discrete_action_to_continuous(discrete_action)

    def random_action(self):
        return self.action_space[np.random.randint(low=0, high=3)]

    def e_greedy_action(self):
        action_rewards = self.dqn.q_network.forward(
            torch.tensor(self.state)
        ).detach().numpy()
        prob = np.random.uniform(low=0.0, high=1.0)
        if prob < self.epsilon:
            return self.random_action()
        else:
            return np.argmax(action_rewards)

    def _discrete_action_to_continuous(self, discrete_action):
        return self.continuous_actions[discrete_action]

    def decrease_episode_length(self, delta=50):
        if self.episode_length > 100:
            self.episode_length -= delta

    def _should_evaluate_policy(self):
        return all([
            self.episode_length == 100,
            self.num_steps_taken > 15000,
            self.episodes_count % 10 == 0,
            self.num_steps_taken % self.episode_length == 0
        ])

    def _load_best_policy_and_weights(self):
        if not self.optimal_policy_loaded and self.best_case.stores_snapshot():
            optimal_weights = self.best_case.get_optimal_weights()
            self.dqn.q_network.load_state_dict(optimal_weights)
            self.optimal_policy_loaded = True
            self.dqn.q_network.eval()


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(
            in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(
            in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(
            self.q_network.parameters(), lr=0.005)
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        # Target network
        self.target_q_network = Network(input_dimension=2, output_dimension=4)
        # to plot loss graph
        self.loss_graph = []
        # Discount factor
        self.discount_factor = 0.9

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()

        # batch = self.replay_buffer.last_entry()
        batch = self.replay_buffer.random_sample()#biased random


        loss = self.compute_long_term_mse_loss(batch)

        loss.backward()

        self.optimiser.step()

        # append to graph
        self.loss_graph.append(loss.item())
        print('loss is: ',loss.item(),'\n')
        # Return the loss as a scalar
        return loss.item()

    def compute_long_term_mse_loss(self, batch):
        s, a, r, s_p, idx = batch
        q_val = self.q_network.forward(torch.tensor(s))

        prediction_tensor = torch.gather(q_val, 1, torch.tensor(a))
        # bellman equation
        q_val_prime = self.target_q_network.forward(torch.tensor(s_p)).detach()
        max_actions = np.argmax(q_val_prime.detach().numpy(), axis=1).reshape(-1, 1)
        s_p_tensor = torch.gather(q_val_prime, 1, torch.tensor(max_actions)).detach()
        expected_value = r + self.discount_factor * s_p_tensor.data.numpy()
        #print((expected_value > np.mean(expected_value, axis=0) + np.std(expected_value, axis=0)).squeeze())
        idx_to_update = idx[(expected_value > np.mean(expected_value, axis=0) + np.std(expected_value, axis=0)).squeeze()]
        #print('idx to update is: ',idx_to_update)
        self.replay_buffer.update_weights(idx_to_update)
        #print(torch.nn.MSELoss()(torch.tensor(expected_value), prediction_tensor))
        return torch.nn.MSELoss()(torch.tensor(expected_value), prediction_tensor)

    def update_target_network(self):
        weights = self.q_network.state_dict()
        self.target_q_network.load_state_dict(weights)

    def show_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_graph)
        plt.yscale('log')
        plt.show()
        return

    def save_model(self):
        torch.save(self.q_network.state_dict(),'C:/Users/SamSung/Desktop/uni/y4/RL/DQN_tutorial/Coursework/cw_section4.pth')
        print('model saved!')
        return

class ReplayBuffer:

    def __init__(self):
        self.buffer = collections.deque(maxlen=10000)
        self.sample_size = 200
        self.sample_weights = collections.deque(maxlen=10000)
        self.min_w = 0.05

    def size(self):
        return len(self.buffer)

    def is_big_enough(self):
        return self.size() >= self.sample_size

    def add(self, transition):
        self.buffer.appendleft(transition)
        self.sample_weights.appendleft(self.min_w)

    def update_weights(self, idx):
        for i in idx:
            self.sample_weights[i] = self.min_w * 2

    def random_sample(self):
        buffer_size = self.size()
        prob = np.array(self.sample_weights)
        prob = prob / np.sum(prob)
        sample_idx = np.random.choice(
            np.arange(buffer_size), size=self.sample_size, replace=False, p=prob)

        states = []
        actions = []
        rewards = []
        states_prime = []

        for idx in sample_idx:
            s, a, r, s_p = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            states_prime.append(s_p)

        #states = np.array(states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float64).reshape(-1, 1)
        actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
        #states_prime = np.array(states_prime, dtype=np.float32)

        return states, actions, rewards, states_prime, sample_idx

    def randomsample2(self):#pure random
        # s a r s_p for loss computing are all np arrays
        states = []
        actions = []
        rewards = []
        states_prime = []

        buffer_size = self.size()
        sample_prob = np.array(self.sample_weights)
        sample_prob = sample_prob / np.sum(sample_prob)
        sample_idx = np.random.choice(np.arange(buffer_size), size=self.batch_size, replace=False, p=sample_prob)

        # create buffer for loss
        for idx in sample_idx:
            s, a, r, s_p = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            states_prime.append(s_p)

        states = np.array(states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float64).reshape(-1, 1)
        actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
        states_prime = np.array(states_prime, dtype=np.float32)

        return states, actions, rewards, states_prime, sample_idx


# Takes care to save the weights when network reaches goal with minimal amount of steps
class save_best_weights:
    def __init__(self):
        self.min_steps_to_goal = 1000  # magic number, assume 100 or less steps in testing
        self.min_distance_to_goal = 1
        self.weights = None

    def save_weights(self, num_steps, weights):
        self.min_steps_to_goal = num_steps
        self.weights = weights

    def weights_without_goal_but_min_dist(self, distance, weights):
        if distance < self.min_distance_to_goal and self.min_steps_to_goal == 1000:
            self.min_distance_to_goal = distance
            self.weights = weights

    def get_optimal_weights(self):
        return self.weights

    def get_min_steps_to_goal(self):
        return self.min_steps_to_goal

    def stores_snapshot(self):
        return self.weights is not None
'''
if __name__ == "__main__":

    environment = Environment(display=True, magnification=500)

    agent = Agent(environment)

    dqn=DQN()

'''
