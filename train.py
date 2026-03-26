import torch
import torch.optim as optim
import torch.nn as nn  # <-- Add this line
import numpy as np
import random
from environment import FightingGameEnv
from dqn import DQN, ReplayBuffer
import matplotlib.pyplot as plt


# --- Training Function ---
def train_dqn(env, episodes=300, batch_size=64, gamma=0.99, lr=1e-3, tau=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    buffer = ReplayBuffer(capacity=10000)
    epsilon, epsilon_decay, epsilon_min = 1.0, 0.995, 0.05

    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = env.action_space.sample() if random.random() < epsilon else policy_net(
                state_tensor).argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                samples = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*samples)
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(np.array(rewards)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(np.array(dones)).to(device)

                q_values = policy_net(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target)  # <-- This line will now work
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Soft update target network
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        winner = "Player" if env.player_health > 0 else "Opponent"
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}, Winner: {winner}")

    # Plot rewards
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()


# --- Main Script ---
if __name__ == "__main__":
    env = FightingGameEnv(render_mode=True)
    train_dqn(env)