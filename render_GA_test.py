import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from pettingzoo.mpe_GA import GA_predator_coalitions_v1
import pygame

def run_and_plot():
    env = GA_predator_coalitions_v1.env(num_predators=3, num_prey=10, render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)

    # Plot benchmark data collected during the episode
    env.unwrapped.scenario.plot_benchmark_data(env.unwrapped.agent_data_history)

if __name__ == "__main__":
    run_and_plot()

