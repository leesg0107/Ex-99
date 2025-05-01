import numpy as np
import gymnasium as gym
from pettingzoo.atari import boxing_v2
import supersuit as ss
from collections import defaultdict
import random

class DynaQAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=5):
        self.Q = defaultdict(lambda: np.zeros(action_space.n))
        self.model = defaultdict(lambda: defaultdict(lambda: (0, 0)))  # (next_state, reward)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.n_planning_steps = n_planning_steps  # number of planning steps
        self.state_space = state_space
        self.action_space = action_space

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        # Update Q-value
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # Update model
        self.model[state][action] = (next_state, reward)

    def planning(self):
        for _ in range(self.n_planning_steps):
            # Sample random state and action from model
            state = random.choice(list(self.model.keys()))
            action = random.choice(list(self.model[state].keys()))
            next_state, reward = self.model[state][action]
            
            # Update Q-value using simulated experience
            best_next_action = np.argmax(self.Q[next_state])
            td_target = reward + self.gamma * self.Q[next_state][best_next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error

def preprocess_observation(obs):
    # Convert observation to a simpler state representation
    # Here we're just taking a simple approach of flattening and discretizing
    return tuple(obs.flatten() // 10)  # Discretize pixel values

def train_dyna_q(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        env.reset()
        state = preprocess_observation(env.observe(env.agent_selection))
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            env.step(action)
            next_state = preprocess_observation(env.observe(env.agent_selection))
            reward = env.rewards[env.agent_selection]
            done = env.terminations[env.agent_selection]

            # Update Q-values and model
            agent.update(state, action, reward, next_state)
            
            # Planning step
            agent.planning()

            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

def main():
    # Create and preprocess the environment
    env = boxing_v2.env(render_mode="rgb_array")
    env = ss.color_reduction_v0(env)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    
    # Reset environment first to get agent selection
    env.reset()
    
    # Initialize agent with the first agent's action space
    agent = DynaQAgent(
        state_space=None,  # Will be determined by preprocessed observation
        action_space=env.action_space(env.agent_selection),
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        n_planning_steps=5
    )
    
    # Train the agent
    train_dyna_q(env, agent, num_episodes=1000)
    
    env.close()

if __name__ == "__main__":
    main() 