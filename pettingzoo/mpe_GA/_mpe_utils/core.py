import numpy as np
import random

class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # reference to individual reward
        self.individual_reward = 0.0
        # Coalition References
        self.coalition = None
        # UBM References
        self.q_vals = {'Q_join': 0.0, 'Q_form': 0.0, 'Q_leave': 0.0}
        self.ubm_last_action = None

class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        # list of all coalitions in the world
        self.coalitions = []
        self.coalition_id_counter = 0  

        self.ubm = None

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

class Coalition: # agent's coalition (team)
    def __init__(self, id, color=None):
        self.id = id
        self.members = []  # List of agents
        self.shared_reward = 0
        self.reward_contributions = []  # List of dicts with agent, reward, distance

        # Assign a random bright RGB color if not specified
        if color is None:
            self.color = np.array([random.uniform(0.5, 1.0) for _ in range(3)])
        else:
            self.color = np.array(color)

    def distribute_rewards(self, coalition):
        """
        Distributes rewards:
        - 50% fixed to catcher(s)
        - 50% shared among coalition members weighted by inverse distance
        """
        if not coalition.reward_contributions:
            return  # No captures to process

        epsilon = 1e-6

        for entry in coalition.reward_contributions:
            catcher = entry['catcher']
            reward = entry['reward']
            distances = entry['distances']  # list of (agent, distance) tuples

            # 1. Give catcher 50%
            catcher_share = reward * 0.5
            if not hasattr(catcher, 'individual_reward'):
                catcher.individual_reward = 0.0
            catcher.individual_reward = catcher_share
            print(f"Catcher {catcher.name} gets fixed share: {catcher_share}")

            # 2. Distribute the remaining 50% based on inverse distance
            shared_pool = reward - catcher_share
            total_weight = 0.0
            weights = []

            for agent, dist in distances:
                weight = 1.0 / (dist + epsilon)
                weights.append((agent, weight))
                total_weight += weight
                print(f"{agent.name} weight: {weight}")

            for agent, weight in weights:
                if total_weight > 0:
                    share = (weight / total_weight) * shared_pool
                else:
                    share = 0.0
                if not hasattr(agent, 'individual_reward'):
                    agent.individual_reward = 0.0
                agent.individual_reward = share
                print(f"\n{agent.name} gets shared pool share: {share} (total now: {agent.individual_reward})")

        # Clear for next capture cycle
        coalition.reward_contributions.clear()

    def decide_coalition_action(self, agent, nearby_agents, world):
        """
        Decide coalition-related action based on observed nearby agents' communication.
        """
        if agent.silent:
            return  # silent agents don't participate in coalition talks

        open_to_join_agents = []
        form_new_agents = []

        for other in nearby_agents:
            if other is agent or other.silent:
                continue
            if np.array_equal(other.state.c, np.array([1, 0, 0])):
                open_to_join_agents.append(other)
            elif np.array_equal(other.state.c, np.array([0, 1, 0])):
                form_new_agents.append(other)

        # 1. Join Coalition
        if len(open_to_join_agents) > 0 and agent.coalition is not None:
            target = open_to_join_agents[0]  # pick the first one for now
            if target.coalition is not None:
                Coalition.join_coalition(target, agent.coalition)      
            return

        # 2. Form New Coalition
        if len(form_new_agents) > 0 and agent.coalition is None:
            partner = form_new_agents[0]
            new_coalition = Coalition.form_coalition(agent, partner, world)
            return

        # 3. Leave Coalition
        if agent.ubm_last_action == "leave" and agent.coalition is not None:
            Coalition.leave_coalition(agent, agent.coalition, world)
            return    
    
    @staticmethod
    def form_coalition(agent1, agent2, world):
        # Only form a coalition if both agents are not already in coalitions
        if agent1.coalition is not None or agent2.coalition is not None:
            print(f"Cannot form coalition: one or both agents are already in a coalition.")
            return None

        coalition = Coalition(world.coalition_id_counter)
        world.coalition_id_counter += 1

        # Add both agents to the coalition
        coalition.members.append(agent1)
        coalition.members.append(agent2)

        # Set coalition reference in agents
        agent1.coalition = coalition
        agent2.coalition = coalition

        world.coalitions.append(coalition)

        print(f"Formed new coalition #{coalition.id} with members: {[agent1.name, agent2.name]}")
        return coalition

    @staticmethod
    def join_coalition(agent, curr_coalition):

        if agent.coalition is not None:
            print(f"{agent.name} is already in a coalition.")
            return False

        curr_coalition.members.append(agent)
        print(f"{agent.name} joined coalition #{curr_coalition.id} with members: {curr_coalition.members}")
        return True

    @staticmethod
    def leave_coalition(agent, curr_coalition, world):

        if agent.coalition is None:
            print(f"{agent.name} is not in coalition #{self.id}.")
            return False

        curr_coalition.members.remove(agent)
        agent.coalition = None
        print(f"{agent.name} left coalition #{curr_coalition.id}")

        if len(curr_coalition.members) <= 1:
            curr_coalition.members[0].coalition =None
            curr_coalition.members.remove(curr_coalition.members[0])
            world.coalitions.remove(curr_coalition)
            print(f"Coalition #{curr_coalition.id} disbanded because it had only one member.")
            # Optional: remove coalition from global list, if any

        return True

class UBM:
    """
    Utility-Based Messaging (UBM) for one agent.
    Messages: 'join'  -> [1,0,0]
              'form'  -> [0,1,0]
              'leave' -> [0,0,1]
              'idle'  -> [0,0,0]

    This is a bandit-style UBM:
    - Each message has an estimated utility Q(msg).
    - decision_rule: ε-greedy, then cost-threshold gating.
    - update: simple incremental update Q <- Q + alpha * (G - Q), where G is observed payoff.
    """

    MESSAGE_VECTORS = {
        "join":  np.array([1, 0, 0], dtype=float),
        "form":  np.array([0, 1, 0], dtype=float),
        "leave": np.array([0, 0, 1], dtype=float),
        "idle":  np.array([0, 0, 0], dtype=float)
    }

    def __init__(self, comm_cost, threshold, epsilon, alpha, gamma):

        self.Comm_cost = comm_cost # cost of sending any message
        self.threshold = threshold # minimum net expected gain to bother sending
        self.epsilon = epsilon     # exploration rate
        self.alpha = alpha         # learning rate
        self.gamma = gamma         # discount factor (optional)

    def decision_rule(agent, ubm):
        """
        Decide communication intent based on Q-values and coalition status.
        Returns communication vector (np.array of length 3).
        """
        
        # If non-predator or silent, stay idle.
        if not getattr(agent, "adversary", True) or getattr(agent, "silent", False):
            return ubm.MESSAGE_VECTORS["idle"], "idle"
        
        # Pull Q-values from agent
        Q_join = agent.q_vals.get('Q_join')
        Q_form = agent.q_vals.get('Q_form')
        Q_leave = agent.q_vals.get('Q_leave')

        # ε-greedy exploration
        if random.random() < ubm.epsilon:
            if agent.coalition is None:   
                options = ["join", "form", "idle"]
            else:
                options = ["leave", "idle"]

            choice = random.choice(options)
            return ubm.MESSAGE_VECTORS[choice], choice
 
        # --- ε-greedy exploitation ---
        if agent.coalition is None:
            max_q = max(Q_form, Q_join)
            if max_q - ubm.Comm_cost > ubm.threshold:                # Broadcast the action with max Q (form or join)
        
                if Q_form >= Q_join:                                 # Broadcast "form coalition" -> [0, 1, 0]            
                    return ubm.MESSAGE_VECTORS["form"], "form"
                else:                                                # Broadcast "join coalition" -> [1, 0, 0]            
                    return ubm.MESSAGE_VECTORS["join"], "join"
            else:
                return ubm.MESSAGE_VECTORS["idle"], "idle"           # Remain idle -> [0, 0, 0]

        else:                                                        # Agent is in coalition
                                                              
            if Q_leave - ubm.Comm_cost > ubm.threshold:              # Broadcast "leave coalition" -> [0, 0, 1]
                return ubm.MESSAGE_VECTORS["leave"], "leave"
            else:                                                    # Remain idle -> [0, 0, 0]
                return ubm.MESSAGE_VECTORS["idle"], "idle"

    def update_q(self, agent, ubm, action_name, reward, next_best_q=0.0, q_learning=False):
        """
        Update the agent's Q for the given action_name.
        - action_name: one of "join","form","leave","idle"
        - reward: scalar reward signal for the communication decision (should account for comm cost if desired)
        - next_best_q (optional): used if q_learning=True to supply max_{a'} Q(s',a')
        - q_learning: if True use TD target reward + gamma*next_best_q, else do bandit incremental update
        """

        if not hasattr(agent, "q_vals"):
            agent.q_vals = {"Q_form": 0.0, "Q_join": 0.0, "Q_leave": 0.0}
        
        q_key = f'Q_{action_name}'
        current_q = agent.q_vals.get(q_key)    

        if current_q is None:
            # nothing to update for idle (or unknown)
            return 

        if q_learning:    # classical TD target (if next_best_q provided)

            td_target = reward + ubm.gamma * next_best_q
            new_q = current_q + ubm.alpha * (td_target - current_q)

        else:            # simple bandit / incremental average style update

            new_q = current_q + ubm.alpha * (reward - current_q)

        agent.q_vals[q_key] = new_q






        





    

    

