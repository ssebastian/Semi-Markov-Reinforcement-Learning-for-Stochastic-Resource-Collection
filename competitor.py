import datetime
import networkx as nx
import numpy as np
from mushroom_rl.algorithms import Agent
from mushroom_rl.core import Core
from mushroom_rl.policy import Policy
from time import time

from environment import StrictResourceTargetTopEnvironment
from main import compute_J
import sys


class Greedy(Policy):

    def __init__(self, env, speed):
        self.speed = speed
        self.env = env
        self.distances = {e[0]: dict(nx.shortest_path_length(env.graph, target=e[0], weight='length'))
                          for e in env.edge_ordering}

    def draw_action(self, state):
        time, position, device_ordering, device_states, violation_times, resource_edges = state
        heuristics = [self._heuristic(state, device) for device in device_ordering]
        best = np.argmax(heuristics)
        action = self.env.edge_ordering.index(resource_edges[device_ordering[best]])
        if sum(np.isfinite(heuristics)) <= 0:
            return [self.env.info.action_space.n-1]
        return [action]

    def _heuristic(self, state, device):
        time, position, device_ordering, device_states, violation_times, resource_edges = state
        if device_states[device] != StrictResourceTargetTopEnvironment.VIOLATION:
            return -np.inf
        if device not in violation_times:
            return -np.inf
        overstayed = time - violation_times[device]
        if overstayed < 0:
            return -np.inf
        distance = self.distances[resource_edges[device][0]][position]
        return -(overstayed + distance / self.speed)


class ACO(Policy):

    def __init__(self, env, ants, computation_time, speed, evaporation_rate=0.1, alpha=1., beta=2, prob_alpha=1800.,
                 max_time=np.inf):
        self.speed = speed
        self.env = env
        self.max_time = max_time
        self.ants = ants
        self.computation_time = computation_time
        self.prob_alpha = prob_alpha
        self.alpha = alpha
        self.beta = beta
        self.default_pheromon = 0.01
        self.evaporation_rate = evaporation_rate
        self.distances = {e[0]: dict(nx.shortest_path_length(env.graph, target=e[0], weight='length'))
                          for e in env.edge_ordering}

    def draw_action(self, state):
        t, position, device_ordering, device_states, violation_times, resource_edges = state
        violations = [d for d in device_ordering if device_states[d] == StrictResourceTargetTopEnvironment.VIOLATION]

        if len(violations) == 0:
            return [self.env.info.action_space.n-1]

        best_solution = None
        best_score = -np.inf
        phero = {}
        self.default_pheromon = 1 / len(violations)

        start = time()
        # while start + self.computation_time > time():
        results = []
        for ant in range(self.ants):
            path, scores = self.run_ant(state, violations, phero)
            score = sum(scores)
            if score >= best_score:
                best_solution = path
                best_score = score
            results.append((path, scores))

        # update pheromones
        # for path, scores in results:
            norm = sum([phero.get(d1, {}).get(d2, self.default_pheromon)
                        for d1 in (position, *violations) for d2 in (position, *violations)])
            for i in range(len(path)):
                d1 = position if i == 0 else path[i-1]
                d2 = path[i]
                old = phero.get(d1, {}).get(d2, self.default_pheromon)
                phero.setdefault(d1, {})[d2] = (1-self.evaporation_rate) * old + scores[i] / norm

            if time() - start > self.computation_time:
                break

        if best_solution is None:
            return [-1]
        else:
            return [self.env.edge_ordering.index(resource_edges[best_solution[0]])]

    def run_ant(self, state, violations, pheromones):
        start, position, device_ordering, device_states, violation_times, resource_edges = state

        scores = []
        path = []
        mask = np.ones(len(violations), dtype=bool)
        pos = position
        t = start

        while sum(mask) > 0:
            travel_times = np.array([self.distances[resource_edges[device][0]][pos] / self.speed
                                     for i, device in enumerate(violations)])
            time_in_violation = np.array([t - violation_times[d] for d in violations])
            assert sum(time_in_violation < 0) == 0

            tag_prob = np.exp(- (travel_times + time_in_violation) / self.prob_alpha)

            phero = np.array([pheromones.get(pos, {}).get(d, self.default_pheromon) for d in violations])
            prob = np.power(phero, self.alpha) + np.power(tag_prob, self.beta)
            prob *= mask
            prob = prob / prob.sum()

            action = np.random.choice(len(violations), p=prob)
            mask[action] = 0
            path.append(violations[action])
            scores.append(tag_prob[action])

            resource_edge = resource_edges[violations[action]]
            t += travel_times[action] + self.env.graph.edges[resource_edge + (0,)]['length'] / self.speed
            pos = resource_edge[1]

            if t - start > self.max_time:
                break

        return path, scores


if __name__ == '__main__':
    speed = 5 / 3.6

    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(time_str)
    params = {
        'seed': 1234,
        # Downtown
        # 'area': ['Courtney', 'Family', 'Victoria Market', 'Regency', 'West Melbourne', 'Princes Theatre', 'Titles',
        #          'Magistrates', 'Supreme', 'Mint', 'Library',
        #          'Hyatt', 'Banks', 'Twin Towers', 'Spencer', 'McKillop', 'Tavistock',
        #          'City Square', 'Chinatown', 'RACV', 'Markilles', 'Degraves'],
        # 'area': 'Docklands',
        'area': 'Queensberry',
        'network': 'GraphConvolutionResourceNetwork',
        'gamma': np.power(0.5, 1./3600.),
        'start_hour': 7,
        'end_hour': 19,
        'initial_replay_size': 5000,
        'max_replay_size': 100000,
        'optimizer': 'rmsprop',
        'learning_rate': 0.00025,
        'decay': .99,
        'batch_size': 32,
        'target_update_frequency': 1000,
        'evaluation_frequency': 10000,
        'average_updates': 8,
        'max_steps': 100000,
        'initial_exploration_rate': 1.,
        'final_exploration_rate': .1,
        'final_exploration_frame': 90000,
        'save': True,
        'cuda': False,
        'name': time_str
    }

    params['train_frequency'] = np.ceil(params['batch_size'] // params['average_updates'])

    observation = lambda env: np.asarray([env.time, env.position, env.device_ordering,
                                          env.spot_states,
                                          env.potential_violation_times, env.resource_edges], dtype=object)
    mdp = StrictResourceTargetTopEnvironment("dataset.db", params['area'],
                                             observation,
                                             gamma=params['gamma'],
                                             allow_wait=True,
                                             start_hour=params['start_hour'],
                                             end_hour=params['end_hour'],
                                             add_time=params['gamma'] < 1,
                                             speed=speed)

    # days = [i for i in range(1, 356) if i % 13 == 1]  # validation
    days = [i for i in range(1, 356) if i % 13 == 0]  # test

    agent = Agent(mdp.info, Greedy(mdp, speed))
    core = Core(agent, mdp)
    value = np.mean(compute_J(core.evaluate(initial_states=days, render=True)))
    print("avg J Greedy", value)
    # mdp.save_rendered("greedy.mp4", 10000)

    t, h = .1, 600.0
    agent = Agent(mdp.info, ACO(mdp, sys.maxsize, t, speed, max_time=h))
    core = Core(agent, mdp)
    value = np.mean(compute_J(core.evaluate(initial_states=days, render=True)))
    print("avg J ACO", t, h, value)
    # mdp.save_rendered("aco_%d_%d.mp4" % (t * 60, h))

    mdp.close()

