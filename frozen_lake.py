import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state_size = env.observation_space.n
action_size = env.action_space.n

# One-hot encoding for discrete states
def one_hot(state):
    vec = np.zeros(state_size)
    vec[state] = 1
    return vec

# Deep Q-Network Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  
        self.epsilon_decay = 0.995  
        self.learning_rate = 0.001  

        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state)).detach()).item()
            
            target_f = self.model(torch.FloatTensor(state)).clone().detach()
            target_f[action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(torch.FloatTensor(state)), target_f)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train the agent
agent = DQNAgent(state_size, action_size)

episodes = 1000
batch_size = 32

for episode in range(episodes):
    state, _ = env.reset()
    state = one_hot(state)
    total_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = one_hot(next_state)

        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size)

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode+1}: Reward {total_reward}, Epsilon {agent.epsilon:.2f}")
            break

# Test the trained agent
state, _ = env.reset()
state = one_hot(state)
env.render()

while True:
    action = agent.act(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    next_state = one_hot(next_state)

    env.render()
    state = next_state

    if done:
        print(f"Final Reward: {reward}")
        break
