import sqlite3
import os
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy
from heapq import heappush, heappop, nsmallest
from tqdm import tqdm

from matplotlib import animation
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import Discrete
import hashlib
import json


class TopEnvironment(Environment):
    """
    Implementation of the travelling officer environment.
    """

    FREE = 0
    OCCUPIED = 1
    VIOLATION = 2
    VISITED = 3

    LOCATION_TABLE = "sensors"
    FULL_JOIN = " " + LOCATION_TABLE + " join restrictions on bay_id=BayID join events on restrictions.DeviceID=events.DeviceId "

    def __init__(self, database, area, observation, delta_degree=0.1, speed=5., gamma=1, start_hour=8, end_hour=22, days_loaded=1, train_days=None, add_time=False, allow_wait=False, project_dir=None):
        # self.area = area
        self.next_event = None
        self.time = None
        self.position = None
        self.done = False
        self.speed = speed
        self.departures = None
        self.spot_states = None
        self.violation_times = None
        self.potential_violation_times = None
        self.allowed_durations = None
        self.delayed_arrival = None
        self.observation = observation
        self.render_images = None
        self.position_plots = None
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.add_time = add_time
        self.days_loaded = days_loaded
        self.days_consumed = np.inf
        self.day = None
        self.end_day = None

        self.train_days = train_days if train_days is not None else list(range(1, 356))

        conn = sqlite3.connect('file:%s?mode=ro' % database, uri=True)
        cursor = conn.cursor()
        if area is None:
            self.areas = []
        elif isinstance(area, list) or isinstance(area, tuple):
            self.areas = area
        else:
            self.areas = [area]
        if len(self.areas) == 0:
            where_areas = " 1 = 1 "  # if no area specified, use all!
        else:
            where_areas = " or ".join(map(lambda a: " Area=\"" + a + "\" ", self.areas))

        cursor.execute("SELECT min(lat) as south, max(lat) as north, min(lon) as west, max(lon) as east "
                       "from devices where " + where_areas)

        row = cursor.fetchone()

        sqlx = 'select DeviceID, ArrivalTime, DepartureTime, duration*60 ' \
               'from events join durations on durations.sign=events.sign ' \
               'where ' + where_areas + ' and "Vehicle Present"="True" order by ArrivalTime'
        self.all_events = pd.read_sql(sqlx, conn, coerce_float=True, params=None)
        self.events = None
        self.event_idx = None

        dx = max(0.01, row[1] - row[0]) * delta_degree
        dy = max(0.01, row[3] - row[2]) * delta_degree
        self.north = row[1] + dx
        self.south = row[0] - dx
        self.west = row[2] - dy
        self.east = row[3] + dy

        if project_dir is None:
            project_dir = os.getcwd()
        data_dir = os.path.join(project_dir, "data")
        area_hash = hashlib.md5(json.dumps(sorted(area)).encode("utf8")).hexdigest()

        graph_file = 'graph_' + area_hash + '.gml'
        try:
            self.graph = ox.load_graphml(graph_file, folder=data_dir)
        except FileNotFoundError:
            self.graph = ox.graph_from_bbox(
                south=self.south, north=self.north, west=self.west, east=self.east, network_type='walk')
            # ox.plot_graph(self.graph)
            ox.save_graphml(self.graph, filename=graph_file, folder=data_dir)
        print("graph nodes: ", len(self.graph.nodes), ", edges: ", len(self.graph.edges))
        self.actions = tuple(self.graph.edges)
        if allow_wait:
            self.actions = self.actions + ('wait', )
        self.edge_indices = {edge: i for i, edge in enumerate(self.actions)}
        self.edge_lengths = nx.get_edge_attributes(self.graph, 'length')

        self.edge_resources = dict()
        self.resource_edges = dict()
        self.devices = dict()
        self.device_ordering = []
        self.spot_plots = dict()

        devices_file = os.path.join(data_dir, "devices_" + area_hash + ".pckl")
        try:
            with open(devices_file, "rb") as f:
                self.devices, self.device_ordering, self.edge_resources, self.resource_edges = pickle.load(f)
        except:
            print("cannot load devices file", devices_file)
            fig, ax = ox.plot_graph(self.graph, show=False, close=False)
            res = cursor.execute("SELECT lat, lon, DeviceID " +
                                 "from devices " +
                                 "where " + where_areas)
            for lat, lon, device_id, *vio_times in tqdm(res.fetchall()):
                # g = ox.truncate_graph_bbox(self.graph, lat+dy, lat-dy, lon+dx, lon-dx, True, True)
                g = ox.truncate_graph_dist(self.graph, ox.get_nearest_node(self.graph, (lat, lon)), 500, retain_all=True)
                nearest = ox.get_nearest_edge(g, (lat, lon))
                edge = (nearest[1], nearest[2])
                self.device_ordering.append(device_id)
                self.devices[device_id] = (lat, lon, edge)
                if edge in self.edge_resources:
                    edge_resources = self.edge_resources[edge]
                else:
                    edge_resources = []
                    self.edge_resources[edge] = edge_resources
                edge_resources.append(device_id)
                self.resource_edges[device_id] = edge
                ax.plot(*nearest[0].xy, color='blue')
                ax.scatter(lon, lat, s=2, c="red")

            with open(devices_file, "wb") as f:
                pickle.dump((self.devices, self.device_ordering, self.edge_resources, self.resource_edges), f)
            plt.show()

        print(len(self.devices), "parking devices")

        mdp_info = MDPInfo(Discrete(2), Discrete(len(self.actions)), gamma, np.inf)
        super().__init__(mdp_info)

        cursor.close()
        conn.close()

    def close(self):
        pass

    def reset(self, state=None):
        year = 2017
        self.done = False
        self.position = min(list(self.graph.nodes))

        if self.days_consumed < self.days_loaded:
            self.time = int(datetime.strptime(("%03d %d %02d:00" % (self.day + self.days_consumed, year, self.start_hour)), "%j %Y %H:%M").strftime("%s"))
            self.end_day = int(datetime.strptime(("%03d %d %02d:00" % (self.day + self.days_consumed, year, self.end_hour)), "%j %Y %H:%M").strftime("%s"))
            self._update_spots()
            return self._state()

        self.delayed_arrival = []
        self.departures = []
        self.violation_times = []
        self.potential_violation_times = dict()
        self.allowed_durations = dict()
        self.spot_states = {id: self.FREE for id in self.devices}
        self.render_images = []

        if state is None:
            self.day = np.random.choice(self.train_days)
        else:
            self.day = int(state)
        start = datetime.strptime(("%03d %d %02d:00" % (self.day, year, self.start_hour)), "%j %Y %H:%M").strftime("%s")
        end = datetime.strptime(("%03d %d %02d:00" % (self.day + (self.days_loaded-1), year, self.end_hour)), "%j %Y %H:%M").strftime("%s")
        self.end_day = int(datetime.strptime(("%03d %d %02d:00" % (self.day, year, self.end_hour)), "%j %Y %H:%M").strftime("%s"))

        self.events = self.all_events.loc[
            (self.all_events['DepartureTime'] > int(start))
            & (self.all_events['ArrivalTime'] < int(end))
        ]
        self.event_idx = 0

        self.days_consumed = 0
        self._next_event()
        self.time = int(start)
        self._update_spots()

        return self._state()

    def _next_event(self):
        while True:
            if self.event_idx + 1 >= len(self.events):
                self.next_event = None
                break

            self.event_idx += 1
            self.next_event = self.events.iloc[self.event_idx]

            if self.next_event[0] in self.devices:
                break

    def _state(self):
        return self.observation(self)

    def _update_spots(self):
        if self.time > self.end_day:
            self.done = True
            self.days_consumed += 1
            return

        while True:
            times = [
                nsmallest(1, self.violation_times)[0][0] if len(self.violation_times) > 0 else np.inf,
                nsmallest(1, self.departures)[0][0] if len(self.departures) > 0 else np.inf,
                nsmallest(1, self.delayed_arrival)[0][0] if len(self.delayed_arrival) > 0 else np.inf,
                self.next_event['ArrivalTime'],
            ]
            if min(times) > self.time:
                break
            arg = np.argmin(times)

        # while self.next_event['ArrivalTime'] <= self.time:
            if arg == 2 or arg == 3:
                if arg == 3:
                    device_id, arrival, departure, duration = self.next_event
                else:  # if arg == 2:
                    t, device_id, arrival, departure, duration = heappop(self.delayed_arrival)
                if departure >= self.time:
                    if self.spot_states[device_id] != self.FREE:
                        t = max([t for t, d in self.departures if d == device_id])
                        if t+1 > departure:
                            heappush(self.delayed_arrival, (t+1, device_id, arrival, departure, duration))
                    else:
                        self.spot_states[device_id] = self.OCCUPIED
                        heappush(self.departures, (departure, device_id))
                        assert sum([d == device_id for t, d in self.departures]) == 1
                        self.potential_violation_times[device_id] = arrival + duration
                        self.allowed_durations[device_id] = duration
                        if departure > arrival + duration:
                            heappush(self.violation_times, (arrival + duration, device_id))
                self._next_event()
                if self.next_event is None or self.next_event['ArrivalTime'] >= self.end_day:
                    self.done = True
                    self.days_consumed += 1
                    break
        # while len(self.violation_times) > 0 and nsmallest(1, self.violation_times)[0][0] <= self.time:
            if arg == 0:
                t, device_id = heappop(self.violation_times)
                self.spot_states[device_id] = self.VIOLATION
                assert device_id in self.potential_violation_times
        # while len(self.departures) > 0 and nsmallest(1, self.departures)[0][0] <= self.time:
            if arg == 1:
                t, device_id = heappop(self.departures)
                self.spot_states[device_id] = self.FREE
                if device_id in self.potential_violation_times:
                    del self.potential_violation_times[device_id]
                    del self.allowed_durations[device_id]
                    assert sum([d == device_id for t, d in self.violation_times]) == 0

            pass

    def legal_actions(self, position=None):
        if position is None:
            position = self.position
        return [self.edge_indices[e + (0,)] for e in self.graph.out_edges(position)]

    def step(self, action):
        edge = self.actions[action[0]]

        reward = 0

        if edge == 'wait':
            travel_time = 10
            # reward += 0.0005
        else:
            if edge[0] != self.position:
                return self._state(), -100, self.done, {}

            self.position = edge[1]
            travel_time = self.edge_lengths[edge] / self.speed

        self.time += travel_time

        self._update_spots()

        if edge != 'wait':
            edge = (edge[0], edge[1])
            if edge in self.edge_resources:
                for device_id in self.edge_resources[edge]:
                    if self.spot_states[device_id] == self.VIOLATION:
                        self.spot_states[device_id] = self.VISITED
                        reward += 1

        if self.add_time:
            r = np.array((reward, travel_time))
        else:
            r = reward

        return self._state(), r, self.done, {}

    def _update_plot(self, plot_entry):
        color_map = {
            self.FREE: "blue",
            self.OCCUPIED: 'green',
            self.VIOLATION: 'red',
            self.VISITED: 'yellow'
        }
        position, spot_states = plot_entry
        for device_id, path in self.spot_plots.items():
            path.set_edgecolor(color_map[spot_states[device_id]])
        features = self.graph.nodes[position]
        self.position_plots.set_offsets([[features['x'], features['y']]])

    def render(self, show=False):
        plot_entry = (self.position, copy.deepcopy(self.spot_states))
        self.render_images.append(plot_entry)
        if show:
            self._update_plot(plot_entry)
            plt.draw()
            plt.pause(0.1)

    def save_rendered(self, file, max_frames=500):
        fig, ax = ox.plot_graph(self.graph, show=False, close=False, node_size=0)

        for id, value in self.devices.items():
            lat, lon, edge = value
            self.spot_plots[id] = ax.scatter(lon, lat, s=2, c="red", zorder=11)
        self.position_plots = ax.scatter(0, 0, s=50, c='blue', zorder=10)

        frames = min(max_frames, len(self.render_images))
        pbar = tqdm(total=frames)

        def update(i):
            self._update_plot(self.render_images[i])
            pbar.update(1)

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=100)

        ani.save(file)

        plt.close(fig)
        pbar.close()


class StrictResourceTargetTopEnvironment(TopEnvironment):

    def __init__(self, *args, allow_wait=False, **kvargs):
        super().__init__(*args, allow_wait=allow_wait, **kvargs)
        action_space = len(self.edge_resources)
        if allow_wait:
            action_space += 1
        self._mdp_info.action_space = Discrete(action_space)
        self.edge_ordering = list(self.edge_resources.keys())
        self.shortest_paths = {}
        for edge in self.resource_edges.values():
            self.shortest_paths[edge] = dict(nx.shortest_path(self.graph, target=edge[0], weight='length'))

    def step(self, action):
        if action[0] == len(self.edge_resources):
            return super().step([len(self.actions)-1])
        edge = self.edge_ordering[action[0]]
        s = None
        rewards = None
        arrived = False
        done = False
        while not arrived and not done:
            path = self.shortest_paths[edge][self.position]
            if len(path) <= 1:
                action = self.edge_indices[edge + (0,)]
                arrived = True
            else:
                next_node = self.shortest_paths[edge][self.position][1]
                action = self.edge_indices[(self.position, next_node, 0)]
            s, r, done, _ = super().step([action])
            self.render(False)
            if rewards is None:
                rewards = r
            else:
                if np.isscalar(r):
                    rewards += r
                else:
                    rewards[1] += r[1]
                    rewards[0] += r[0] * self._mdp_info.gamma ** r[1]
            if done:
                break
        return s, rewards, done, {}

