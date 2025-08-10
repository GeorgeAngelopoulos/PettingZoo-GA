import numpy as np
from gymnasium.utils import EzPickle
import matplotlib.pyplot as plt
import math

# Import from your local custom MPE copy
from .._mpe_utils.core import Agent, Landmark, World, Coalition, UBM
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

class CONSTANTS:

   # Color Arrays
   PREDATOR_COLOR = [0.85, 0.35, 0.35]
   PREY_COLOR = [0.35, 0.85, 0.35]
   CAUGHT_COLOR = [0.5, 0.5, 0.5]


# Main environment class
class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_predators=3,
        num_prey=10,
        local_ratio=1,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        self.continuous_actions = continuous_actions  # Needed by SimpleEnv
        
        # Create scenario and world with specified number of predators and prey
        self.scenario = Scenario()
        self.world = self.scenario.make_world(num_predators, num_prey)
        
        # Call SimpleEnv init with all required params
        super().__init__(
            scenario=self.scenario,
            world=self.world,
            max_cycles=max_cycles,
            render_mode=render_mode,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
            dynamic_rescaling=dynamic_rescaling,
        )
        
        EzPickle.__init__(
            self,
            num_predators=num_predators,
            num_prey=num_prey,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=dynamic_rescaling,
        )

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

# Scenario logic
class Scenario(BaseScenario):

    def make_world(self, num_predators, num_prey):
        world = World()
        world.dim_c = 3
        world.collaborative = True
        world.coalitions = []
        
        world.ubm = UBM(comm_cost=0.1, threshold=0.05, epsilon=0.9, alpha=0.05, gamma=0.0)

        # Create predators who CAN communicate
        predators = [Agent() for i in range(num_predators)]
        for i, pred in enumerate(predators):
            pred.name = f"Predator_{i}"
            pred.collide = True
            pred.silent = False
            pred.size = 0.06
            pred.adversary = True
            pred.coalition = None

        # Create prey who CANNOT communicate
        preys = [Agent() for i in range(num_prey)]
        for i, prey in enumerate(preys):
            prey.name = f"Prey_{i}"
            prey.collide = True
            prey.silent = True
            prey.size = 0.12
            prey.adversary = False
            prey.caught = False

        world.agents = predators + preys
        world.predators = predators
        world.preys = preys

        return world
# -----------------------------------------------------------------------------------------------------------------------
    def reset_world(self, world, np_random):

        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

        # Agent's colors
        for predator in world.predators:
            predator.color = np.array(CONSTANTS.PREDATOR_COLOR)
        for prey in world.preys:
            prey.color = np.array(CONSTANTS.PREY_COLOR)

        # Reset communication state to zero vector on each reset
        agent.state.c = np.zeros(world.dim_c)
# -----------------------------------------------------------------------------------------------------------------------
    def observation(self, agent, world, comm_range=1):
        obs = []
        # Own velocity and position
        obs.append(agent.state.p_vel)
        obs.append(agent.state.p_pos)

        comm = []
        relative_positions = []

        for other in world.agents:
            if other is agent:
                continue
            # Calculate distance
            dist = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
            if dist <= comm_range:
                
                # Add relative position only if within communication range
                relative_positions.append(other.state.p_pos - agent.state.p_pos)

                # Add communication vector if available
                if not other.silent and isinstance(other.state.c, np.ndarray):
                    comm.append(other.state.c)
                else:
                    comm.append(np.zeros(world.dim_c))

        # Flatten and append relative positions
        if relative_positions:
            obs.append(np.concatenate(relative_positions))
        else:
            obs.append(np.zeros(world.dim_p * 0))  # or zeros with fixed size if you prefer

        # Flatten and append communication
        if comm:
            obs.append(np.concatenate(comm))
        else:
            obs.append(np.zeros(world.dim_c * 0))  # or zeros with fixed size if you prefer

        return np.concatenate(obs)
# -----------------------------------------------------------------------------------------------------------------------
    def is_collision(self, agent1, agent2):

        if agent1 is agent2:
            return False  # No self-collision

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        distance = np.linalg.norm(delta_pos)
        min_distance = agent1.size + agent2.size

        return distance < min_distance*2
# -----------------------------------------------------------------------------------------------------------------------
    def reward(self, agent, world):
        """
        Computes the personal rewards for different collision scenarios of Predator and Prey.
        
        If the *Predator* is not in coalition -> Personal Reward + 1
        If the *Predator* is in coalition -> Personal Reward = 0 and logs his contribution to the coalition shared reward system
        If the *Prey* collides with Predator -> Personal Reward - 1
        
        Parameters:
            agent: Each agent in the scenario.
            world: The world of the scenario.
        """
        reward = 0

        if agent.adversary:
            for prey in world.preys:
                if prey.caught:
                    continue

            if self.is_collision(agent, prey):
                reward = 1
                print(f"SCEN: {agent.name} CAUGHT {prey.name}")

                if agent.coalition is not None:
                    print(f"SCEN: {agent.name} in coalition")
                    distances = []
                    # Loop through all coalition members except the catcher
                    for other_agent in agent.coalition.members:
                        if other_agent is not agent and getattr(other_agent, 'adversary', False):
                            dist = np.linalg.norm(other_agent.state.p_pos - prey.state.p_pos)
                            distances.append((other_agent, dist))  # tuple: (agent, distance)
                            print(f"SCEN: Distance to prey {other_agent.name},{dist}")
                    
                    # Append to coalition's reward_contributions
                    agent.coalition.reward_contributions.append({
                        'catcher': agent,       # renamed from 'agent'
                        'reward': reward,
                        'distances': distances  # list of (agent, distance) tuples
                    })

                    print(f"reward_contributions: {agent.coalition.reward_contributions}")

        else:
            for predator in world.predators:
                if self.is_collision(agent, predator):
                    reward -= 1

        if agent.adversary and agent.coalition is not None:
            return agent.individual_reward

        return reward
# -----------------------------------------------------------------------------------------------------------------------
    def global_reward(self, world):
        return 0
# -----------------------------------------------------------------------------------------------------------------------
    def benchmark_data(self, agent, world, rewardEX):
        """
        Returns benchmark info per agent for the current step:
        - reward: agent's personal reward this step
        - caught_prey: bool, True if agent collided with a prey this step
        - in_coalition: bool, True if agent is in a coalition
        """

        # Individual reward this step from the reward function
        reward = self.reward(agent, world)

        # Check if agent caught a prey this step
        caught_prey = False
        if agent.adversary:  # assuming adversary == predator
            if reward > 0:
                caught_prey = True

        # Check coalition membership
        in_coalition = agent.coalition is not None

        return {
            "reward": reward,
            "caught_prey": caught_prey,
            "in_coalition": in_coalition
        }
# -----------------------------------------------------------------------------------------------------------------------
    def plot_benchmark_data(self, agent_data_history):
        """
        agent_data_history: dict
        Keys: agent names
        Values: list of dicts with keys: 'reward', 'caught_prey', 'in_coalition' per step

        Shows a new figure for every 3 agents, each with 3 subplots per agent.
        """

        agents = list(agent_data_history.items())
        agents_per_figure = 3
        num_figures = math.ceil(len(agents) / agents_per_figure)

        for fig_idx in range(num_figures):
            fig_agents = agents[fig_idx * agents_per_figure : (fig_idx + 1) * agents_per_figure]
            fig, axes = plt.subplots(len(fig_agents), 3, figsize=(15, 4 * len(fig_agents)), squeeze=False)

            for i, (agent_name, data_list) in enumerate(fig_agents):
                steps = list(range(len(data_list)))

                rewards = [d['reward'] for d in data_list]
                caught_prey = [1 if d['caught_prey'] else 0 for d in data_list]
                in_coalition = [1 if d['in_coalition'] else 0 for d in data_list]

                # Reward plot
                axes[i, 0].plot(steps, rewards)
                axes[i, 0].set_title(f'{agent_name} - Reward per Step')
                axes[i, 0].set_xlabel('Step')
                axes[i, 0].set_ylabel('Reward')

                # Caught prey plot
                axes[i, 1].scatter(steps, caught_prey, color='red', marker='o')
                axes[i, 1].set_title(f'{agent_name} - Caught Prey')
                axes[i, 1].set_xlabel('Step')
                axes[i, 1].set_yticks([0, 1])
                axes[i, 1].set_yticklabels(['No', 'Yes'])

                # In coalition plot
                axes[i, 2].scatter(steps, in_coalition, color='green', marker='x')
                axes[i, 2].set_title(f'{agent_name} - In Coalition')
                axes[i, 2].set_xlabel('Step')
                axes[i, 2].set_yticks([0, 1])
                axes[i, 2].set_yticklabels(['No', 'Yes'])

            plt.tight_layout()
            plt.show()
# -----------------------------------------------------------------------------------------------------------------------
