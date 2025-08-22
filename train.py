import pygame
import numpy as np
from environment import GridWorld
from dqn_agent import DQNAgent

pygame.init()

cell_size = 40
env = GridWorld(size=10, num_walls=20)
agent = DQNAgent(state_shape=(10, 10), action_size=4)

window = pygame.display.set_mode((cell_size * env.size, cell_size * env.size + 50))
pygame.display.set_caption("DQN Navigation Game")
font = pygame.font.SysFont("Arial", 24)

colors = {
    0: (255, 255, 255),  # Empty
    1: (0, 0, 0),        # Wall
    2: (0, 0, 255),      # Agent (Blue)
    3: (0, 255, 0),      # Goal (Green)
}

def draw_grid(grid, score=0, episode=0):
    # Draw the grid cells
    for i in range(env.size):
        for j in range(env.size):
            val = grid[i, j]
            pygame.draw.rect(window, colors.get(val, (255, 0, 0)), (j*cell_size, i*cell_size, cell_size, cell_size))
            pygame.draw.rect(window, (180, 180, 180), (j*cell_size, i*cell_size, cell_size, cell_size), 1)

    # Draw the agent (blue)
    i, j = env.agent_pos
    pygame.draw.rect(window, (0, 0, 255), (j*cell_size, i*cell_size, cell_size, cell_size))  # Agent (Blue)

    # Draw the goal (green)
    goal_i, goal_j = env.goal_pos
    pygame.draw.rect(window, (0, 255, 0), (goal_j*cell_size, goal_i*cell_size, cell_size, cell_size))  # Goal (Green)

    # Draw score and episode info at the bottom
    pygame.draw.rect(window, (50, 50, 50), (0, env.size * cell_size, cell_size * env.size, 50))
    text = font.render(f"Score: {score}  Episode: {episode}", True, (255, 255, 255))
    window.blit(text, (10, env.size * cell_size + 10))

    pygame.display.update()

clock = pygame.time.Clock()
episodes = 500
for ep in range(1, episodes + 1):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)  # Normally the agent would act

        # Handle manual control via keyboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # Move up
                elif event.key == pygame.K_DOWN:
                    action = 1  # Move down
                elif event.key == pygame.K_LEFT:
                    action = 2  # Move left
                elif event.key == pygame.K_RIGHT:
                    action = 3  # Move right

        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

        draw_grid(env.grid, score=int(total_reward), episode=ep)

        clock.tick(5)  # Slower movement for better observation

pygame.quit()
