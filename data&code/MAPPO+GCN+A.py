import math
import random
random.seed(a=123)
import torch
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
import torch.nn as nn
from pylab import *
from torch_geometric.data import Data

import numpy as np

np.random.seed(123)
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
import pandas as pd
import traci
import xml.etree.ElementTree as ET
import networkx as nx
from torch.distributions import Categorical

device = torch.device("cuda:0")

file_path = 'railway.net.xml'
tree = ET.parse(file_path)
root = tree.getroot()
G = nx.DiGraph()

charging_stations_details = {
    "chargingStation1": {"jid": "9098567743", "availableChargers": 4, "chargingPower": 120, "stationId": 1},
    "chargingStation2": {"jid": "3922520339", "availableChargers": 6, "chargingPower": 120, "stationId": 2},
    "chargingStation3": {"jid": "4532946549", "availableChargers": 6, "chargingPower": 120, "stationId": 3},
    "chargingStation4": {"jid": "1192622508", "availableChargers": 4, "chargingPower": 120, "stationId": 4},
    "chargingStation5": {"jid": "5469345620", "availableChargers": 10, "chargingPower": 60, "stationId": 5},
    "chargingStation6": {"jid": "11071870503", "availableChargers": 12, "chargingPower": 60, "stationId": 6},
    "chargingStation7": {"jid": "3933518870", "availableChargers": 8, "chargingPower": 120, "stationId": 7},
    "chargingStation8": {"jid": "4530980711", "availableChargers": 6, "chargingPower": 90, "stationId": 8},
    "chargingStation9": {"jid": "1965559133", "availableChargers": 6, "chargingPower": 90, "stationId": 9},
    "chargingStation10": {"jid": "4097553150", "availableChargers": 20, "chargingPower": 60, "stationId": 10},
    "chargingStation11": {"jid": "1965559026", "availableChargers": 6, "chargingPower": 120, "stationId": 11},
    "chargingStation12": {"jid": "11071870572", "availableChargers": 6, "chargingPower": 180, "stationId": 12},
    "chargingStation13": {"jid": "11009827460", "availableChargers": 4, "chargingPower": 160, "stationId": 13},
    "chargingStation14": {"jid": "8380501943", "availableChargers": 5, "chargingPower": 90, "stationId": 14}
}

charging_stations = {
    "chargingStation1": "-984015955#0",
    "chargingStation2": "902243070#2",
    "chargingStation3": "389390029#1",
    "chargingStation4": "103279745",
    "chargingStation5": "568515711",
    "chargingStation6": "1192735216#0",
    "chargingStation7": "186030204#2",
    "chargingStation8": "307005415#0",
    "chargingStation9": "-186361961#1",
    "chargingStation10": "902243070#0",
    "chargingStation11": "185866706#9",
    "chargingStation12": "-186362527#1",
    "chargingStation13": "1028176908#2",
    "chargingStation14": "389389609#1"
}

# Build graph nodes from SUMO junctions.
for junction in root.findall('.//junction'):
    jid = junction.get('id')
    x = junction.get('x')
    y = junction.get('y')
    csid = 0  # Zero denotes a junction without a charging station.
    for station, info in charging_stations_details.items():
        if info['jid'] == jid:
            csid = info['stationId'] / 14
            break
    if x is not None and y is not None:
        G.add_node(jid, pos=(float(x) / 7100, float(y) / 12000), csid=csid)

# Build road edges using mean lane speeds and lengths.
for edge in root.findall('.//edge'):
    eid = edge.get('id')
    from_junction = edge.get('from')
    to_junction = edge.get('to')
    if from_junction and to_junction:
        total_speed = 0.0
        total_length = 0.0
        lane_count = 0
        for lane in edge.findall('lane'):
            speed = lane.get('speed')
            length = lane.get('length')
            if speed and length:
                total_speed += float(speed)
                total_length += float(length)
                lane_count += 1
        if lane_count > 0:
            average_speed = total_speed / lane_count
            average_length = total_length / lane_count
        else:
            average_speed = 0.0
            average_length = 0.0
        G.add_edge(from_junction, to_junction, id=eid, speed=average_speed, length=average_length)

# Map junction IDs to tensor indices.
node_id_map = {nid: idx for idx, nid in enumerate(G.nodes())}
node_features = torch.tensor([[float(G.nodes[n]['pos'][0]), float(G.nodes[n]['pos'][1]), float(G.nodes[n]['csid'])] for n in G.nodes()], dtype=torch.float32)
edge_index = torch.tensor([[node_id_map[u], node_id_map[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
edge_weight = torch.tensor([float(G.edges[e]['length']) / float(G.edges[e]['speed']) for e in G.edges()], dtype=torch.float32)
edge_weight = torch.exp(-edge_weight / 10)
Graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
cs_node_id_list = [jid for jid in G.nodes() if G.nodes[jid]['csid'] != 0]
cs_idx = [node_id_map[jid] for jid in cs_node_id_list]
cs_idx = torch.tensor(cs_idx, dtype=torch.long, device=node_features.device)
num_nodes = G.number_of_nodes()
mask = torch.isin(torch.arange(num_nodes), cs_idx)   # True for charging station nodes.
Graph.cs_mask = mask


def Graphupdate():
    for edge_id in G.edges():
        mean_speed = traci.edge.getLastStepMeanSpeed(G.edges[edge_id]['id'])
        safe_speed = max(mean_speed, 0.1)
        G.edges[edge_id]['speed'] = safe_speed
    edge_weight = torch.tensor([float(G.edges[e]['length']) / float(G.edges[e]['speed']) for e in G.edges()], dtype=torch.float32)
    edge_weight = torch.exp(-edge_weight / 10)
    Graph.edge_attr = edge_weight


class PPOBuffer:
    def __init__(self):
        self.buffer = []
    def clear(self):
        self.buffer = []
    # Store a decision before its delayed reward is available.
    def pre_add(self, state, action, log_prob, value, time_step):
        index = len(self.buffer)
        transition = {
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'time_step': time_step,
            'reward': None,
            'finished': False
        }
        self.buffer.append(transition)
        return index
    # Fill in the reward when charging completes.
    def update_reward(self, index, reward):
        if index < len(self.buffer):
            self.buffer[index]['reward'] = reward
            self.buffer[index]['finished'] = True
    def finalize_trajectory(self):
        # Keep completed requests; filtering may introduce gaps in the trajectory.
        valid_buffer = [t for t in self.buffer if t['finished']]
        # Compute elapsed time between consecutive retained decisions for GAE.
        for i in range(len(valid_buffer) - 1):
            t_current = valid_buffer[i]['time_step']
            t_next = valid_buffer[i + 1]['time_step']
            valid_buffer[i]['duration'] = max(0, t_next - t_current)
        # The final decision has no successor.
        if len(valid_buffer) > 0:
            valid_buffer[-1]['duration'] = 0.0
        self.buffer = valid_buffer

    def get_batch(self):
        states = [t['state'] for t in self.buffer]
        actions = torch.tensor([t['action'] for t in self.buffer], dtype=torch.long)
        log_probs = torch.tensor([t['log_prob'] for t in self.buffer], dtype=torch.float32)
        rewards = torch.tensor([t['reward'] for t in self.buffer], dtype=torch.float32)
        values = torch.tensor([t['value'] for t in self.buffer], dtype=torch.float32)
        durations = torch.tensor([t['duration'] for t in self.buffer], dtype=torch.float32)

        return states, actions, log_probs, rewards, values, durations

    def __len__(self):
        return len(self.buffer)

class StationAttention(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=4, q_input=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # Query: vehicle features; keys and values: 8 GCN + 5 station features.
        self.query = nn.Linear(q_input, hidden_dim)
        self.key = nn.Linear(13, hidden_dim)
        self.value = nn.Linear(13, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 13)
        self.layer_norm = nn.LayerNorm(13)
        self.last_attention_weights = None

    def forward(self, vi_expanded, cs_features, csi):
        """
        cs_features: [batch, num_stations, 8] GCN embeddings.
        csi: [batch, num_stations, 5] real-time station features.
        """
        batch_size, num_stations, _ = cs_features.shape

        station_kv = torch.cat([cs_features, csi], dim=-1)

        Q = self.query(vi_expanded)
        K = self.key(station_kv)
        V = self.value(station_kv)

        # Split features into attention heads.
        Q = Q.view(batch_size, num_stations, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_stations, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_stations, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_stations, -1)
        attended = self.out_proj(attended)
        # Preserve map and station features through the residual connection.
        output = self.layer_norm(station_kv + attended)
        return output

nearest_5 = [[1, 6, 12, 2, 4],
             [6, 0, 9, 4, 13],
             [3, 10, 11, 5, 0],
             [10, 2, 5, 11, 8],
             [7, 1, 13, 8, 5],
             [10, 3, 8, 2, 4],
             [1, 9, 0, 13, 4],
             [4, 13, 9, 1, 8],
             [5, 10, 4, 3, 7],
             [13, 6, 1, 7, 4],
             [5, 3, 2, 8, 11],
             [2, 3, 10, 5, 12],
             [0, 2, 1, 6, 3],
             [9, 7, 4, 1, 6]]

class SharedPolicyNet(nn.Module):
    """
    Shared actor for all charging stations.

    The GCN, attention block, and MLP are shared across stations. A learned
    station embedding and station-specific local-neighborhood features retain
    the identity and spatial differences of each charging station.
    """
    def __init__(self, num_features, num_stations, station_embedding_dim=8):
        super(SharedPolicyNet, self).__init__()
        if num_stations != len(nearest_5):
            raise ValueError(
                f"SharedPolicyNet expects {len(nearest_5)} stations, "
                f"but received {num_stations}."
            )

        self.conv1 = GCNConv(num_features, 6)
        self.conv2 = GCNConv(6, 8)
        self.v_cs_attention = StationAttention(hidden_dim=32, num_heads=4)
        self.num_stations = num_stations
        self.station_embedding = nn.Embedding(
            num_stations, station_embedding_dim
        )

        # The same scoring MLP is applied to every candidate station.
        self.fc1 = nn.Linear(
            6 * 13 + 8 + 4 + 5 + 1 + station_embedding_dim,
            128
        )
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        candidate_indices = torch.tensor(
            [[station_id] + nearest_5[station_id]
             for station_id in range(num_stations)],
            dtype=torch.long
        )
        self.register_buffer("candidate_indices", candidate_indices)

    def forward(self, state):
        # A single state is [Graph, vehicle_state, station_state, avg_action].
        # A minibatch is a list of such states.
        if isinstance(state, (tuple, list)) and len(state) == 4:
            if isinstance(state[0], Data):
                state = [state]

        model_device = next(self.parameters()).device
        G_list, vi_list, csi_list, aver_action_list = zip(*state)
        batch_G = Batch.from_data_list(G_list).to(model_device)
        vi = torch.stack(vi_list, dim=0).to(model_device)
        csi = torch.stack(csi_list, dim=0).to(model_device)
        aver_action = torch.stack(aver_action_list, dim=0).to(model_device)

        node_features = F.relu(
            self.conv1(batch_G.x, batch_G.edge_index, batch_G.edge_attr)
        )
        node_features = F.relu(
            self.conv2(node_features, batch_G.edge_index, batch_G.edge_attr)
        )

        batch_size = vi.shape[0]
        cs_features = node_features[batch_G.cs_mask].view(
            batch_size, self.num_stations, 8
        )
        csi_features = csi.view(batch_size, self.num_stations, 5)

        # Construct one local six-station neighborhood for each candidate.
        local_cs_features = cs_features[:, self.candidate_indices, :]
        local_csi_features = csi_features[:, self.candidate_indices, :]

        local_batch_size = batch_size * self.num_stations
        local_cs_features = local_cs_features.reshape(
            local_batch_size, 6, 8
        )
        local_csi_features = local_csi_features.reshape(
            local_batch_size, 6, 5
        )
        vi_expanded = vi.unsqueeze(1).unsqueeze(2).expand(
            -1, self.num_stations, 6, -1
        ).reshape(local_batch_size, 6, 4)

        attended_features = self.v_cs_attention(
            vi_expanded, local_cs_features, local_csi_features
        )
        attended_features = attended_features.reshape(
            batch_size, self.num_stations, 6 * 13
        )

        station_ids = torch.arange(
            self.num_stations, device=model_device
        ).unsqueeze(0).expand(batch_size, -1)
        station_embeddings = self.station_embedding(station_ids)

        vehicle_features = vi.unsqueeze(1).expand(
            -1, self.num_stations, -1
        )
        candidate_features = torch.cat(
            (
                attended_features,
                cs_features,
                vehicle_features,
                csi_features,
                aver_action[:, :self.num_stations].unsqueeze(-1),
                station_embeddings,
            ),
            dim=-1,
        )

        candidate_features = candidate_features.reshape(
            batch_size * self.num_stations, -1
        )
        x = F.relu(self.fc1(candidate_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x).reshape(batch_size, self.num_stations)

class ValueNet(nn.Module):
    def __init__(self, num_features):
        super(ValueNet, self).__init__()
        self.conv1 = GCNConv(num_features, 6)
        self.conv2 = GCNConv(6, 8)
        self.station_attention = StationAttention(hidden_dim=32, num_heads=4)
        self.fc1 = nn.Linear(14 * 13 + 14 * 8 + 4 + 14 * 5 + 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        if next(self.parameters()).device != torch.device("cuda:0"):
            state = [state]
        G_list, vi_list, csi_list, aver_action_list = zip(*state)
        batch_G = Batch.from_data_list(G_list).to(next(self.parameters()).device)
        vi = torch.stack(vi_list, dim=0).to(next(self.parameters()).device)
        csi = torch.stack(csi_list, dim=0).to(next(self.parameters()).device)
        aver_action = torch.stack(aver_action_list, dim=0).to(next(self.parameters()).device)

        node_features = F.relu(self.conv1(batch_G.x, batch_G.edge_index, batch_G.edge_attr))
        node_features = F.relu(self.conv2(node_features, batch_G.edge_index, batch_G.edge_attr))

        # Extract charging station embeddings for global attention.
        batch_size = vi.shape[0]
        cs_features = node_features[batch_G.cs_mask].view(batch_size, 14, 8)
        csi_features = csi.view(batch_size, 14, 5)
        vi_expanded = vi.unsqueeze(1).expand(-1, 14, -1)
        attended_features = self.station_attention(vi_expanded, cs_features, csi_features)
        flattened_features = attended_features.flatten(start_dim=1)

        # Combine attention output, raw features, and the average action.
        combined_features = torch.cat((flattened_features, cs_features.view(-1, 14 * 8), vi, csi, aver_action), dim=1)

        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MAPPO:
    def __init__(self, num_features, num_agents, actor_lr, critic_lr, gamma, eps_clip, k_epochs, lam):
        self.num_agents = num_agents
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = SharedPolicyNet(num_features, num_agents)
        # A centralized critic estimates the global state value.
        self.critic = ValueNet(num_features)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.MseLoss = nn.MSELoss()

        self.critic_loss_list = []

    def get_joint_logits(self, state):
        return self.actor(state)

    def take_action(self, state, deterministic=False):
        with torch.no_grad():
            joint_logits = self.get_joint_logits(state)
            dist = Categorical(logits=joint_logits)
            if deterministic:
                action = torch.argmax(joint_logits, dim=-1)
            else:
                action = dist.sample()
            action_log_prob = dist.log_prob(action)
            value = self.critic(state)
        return action.item(), action_log_prob.item(), value.item()

    def update(self, buffer, batch_size=64):
        self.actor.to(self.device)
        self.critic.to(self.device)

        old_states, old_actions, old_log_probs, rewards, old_values, durations = buffer.get_batch()

        # SMDP GAE assumes decisions are stored in chronological order.
        advantages = []
        gae = 0
        # Use zero as the value following the final transition.
        values = old_values.detach().numpy()
        values = np.append(values, 0)
        rewards = rewards.numpy()
        durations = durations.numpy()

        for t in reversed(range(len(rewards))):
            # Discount by elapsed time between decisions.
            discount = self.gamma ** durations[t]
            delta = rewards[t] + discount * values[t + 1] - values[t]
            gae = delta + discount * self.lam * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        # Construct critic targets before normalizing policy advantages.
        returns = advantages + old_values.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        total_samples = len(old_states)
        epoch_critic_loss = 0
        update_counts = 0

        # Optimize PPO over shuffled minibatches.
        for _ in range(self.k_epochs):
            indices = np.arange(total_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, total_samples, batch_size):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_states = [old_states[i] for i in batch_indices]
                batch_actions = old_actions[batch_indices].to(self.device)
                batch_log_probs = old_log_probs[batch_indices].to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)
                batch_returns = returns[batch_indices].to(self.device)

                joint_logits = self.get_joint_logits(batch_states)
                new_values = self.critic(batch_states).squeeze()

                dist = Categorical(logits=joint_logits)
                new_log_probs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                # Combine the clipped policy objective, entropy bonus, and value loss.
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy
                critic_loss = self.MseLoss(new_values, batch_returns)
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                epoch_critic_loss += critic_loss.item()
                update_counts += 1

        if update_counts > 0:
            self.critic_loss_list.append(epoch_critic_loss / update_counts)
        else:
            self.critic_loss_list.append(0)

        buffer.clear()
        self.actor.to('cpu')
        self.critic.to('cpu')

    def save(self):
        torch.save(self.actor.state_dict(), 'mappo_actor_shared.pth')
        torch.save(self.critic.state_dict(), 'mappo_critic.pth')

    def load(self):
        self.actor.load_state_dict(
            torch.load('mappo_actor_shared.pth', map_location='cpu')
        )
        self.critic.load_state_dict(
            torch.load('mappo_critic.pth', map_location='cpu')
        )

    def lr_change(self, episode):
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_lr * 0.99 ** episode
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_lr * 0.99 ** episode


actor_lr = 5e-4
critic_lr = 5e-4
num_episodes = 200
gamma = 0.9
lam = 0.95
eps_clip = 0.2
k_epochs = 5
num_features = 3
battery_capacity = 46000
battery_level_threshold = 0.2
replay_buffer = PPOBuffer()


def train_mappo_agent(agent, num_episodes, replay_buffer):
    return_list = []
    avg_critic_loss_list = []
    maxre = -100000
    route_files = ["train.rou.xml", "train1.rou.xml", "train2.rou.xml", "train3.rou.xml", "train4.rou.xml"]
    last_route_files = "train.rou.xml"
    for i in range(num_episodes):
        episode_return = 0
        succ_num = 0
        for rou in range(5):
            recommended_vehicles = {}
            selected_route_file = route_files[rou]
            with open("train.sumocfg", "r") as f:
                config_content = f.read()
            new_config_content = config_content.replace(last_route_files, selected_route_file)
            with open("train.sumocfg", "w") as f:
                f.write(new_config_content)
            print('last_route_files', last_route_files)
            print('selected_route_file', selected_route_file)
            last_route_files = selected_route_file
            traci.start(["E:/SUMO/bin/sumo.exe", "-c", "train.sumocfg", '--start'])
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep(traci.simulation.getTime() + 10)
                time = traci.simulation.getTime()
                Graphupdate()
                if time % 100 == 0:
                    print('当前轮次为', i + 1 + rou/10)
                    print('当前时间为', time)
                for veh_id in traci.vehicle.getIDList():
                    battery_level = float(traci.vehicle.getParameter(veh_id, "device.battery.chargeLevel"))
                    if veh_id not in recommended_vehicles:
                        if battery_level < battery_level_threshold * battery_capacity:
                            print(f"Vehicle {veh_id} needs charging. Current battery level: {battery_level}")
                            pos = traci.vehicle.getPosition(veh_id)
                            x_normalized = pos[0] / 7100
                            y_normalized = pos[1] / 12000
                            angle = traci.vehicle.getAngle(veh_id) / 360
                            battery_level_normalized = battery_level / battery_capacity
                            current_edge = traci.vehicle.getRoadID(veh_id)
                            csi = []
                            for station, info in charging_stations_details.items():
                                parking_area_id = f"parkingArea{list(charging_stations.keys()).index(station) + 1}"
                                parknum = traci.parkingarea.getVehicleCount(parking_area_id)
                                availableChargingPoints = info['availableChargers'] - parknum
                                route = traci.simulation.findRoute(current_edge, charging_stations[station], routingMode=1)
                                csi.append(availableChargingPoints / 20)
                                csi.append(info['availableChargers'] / 20)
                                csi.append(info['chargingPower'] / 180)
                                csi.append(min(route.length / 16000, 1))
                                csi.append(min(route.travelTime / 3600, 1))
                            if recommended_vehicles == {}:
                                aver_action = np.zeros(14)
                                small_indices = np.array([])
                            else:
                                acs = [veh_info['action_idx'] for veh_info in recommended_vehicles.values()]
                                one_hot_vectors = []
                                for ac in acs:
                                    one_hot = np.eye(14)[ac]
                                    one_hot_vectors.append(one_hot)
                                one_hot_matrix = np.array(one_hot_vectors)
                                aver_action = np.mean(one_hot_matrix, axis=0)
                                small_indices = np.where(aver_action < 1 / 14)[0]
                            state = [Graph, torch.tensor([x_normalized, y_normalized, battery_level_normalized, angle], dtype=torch.float32),
                                     torch.tensor(csi, dtype=torch.float32), torch.tensor(aver_action, dtype=torch.float32)]
                            action_idx, log_prob, val = agent.take_action(state)
                            charging_station_name = list(charging_stations.keys())[action_idx]
                            charging_station_edge = charging_stations[charging_station_name]
                            print(f"Recommending charging station {charging_station_name} to vehicle {veh_id}.")
                            route = traci.simulation.findRoute(current_edge, charging_station_edge, routingMode=1)
                            edge_list = route.edges
                            print(f"Edge list for vehicle {veh_id}: {edge_list}")
                            try:
                                traci.vehicle.setRoute(veh_id, edge_list)
                                parking_area_id = f"parkingArea{list(charging_stations.keys()).index(charging_station_name) + 1}"
                                traci.vehicle.setParkingAreaStop(veh_id, parking_area_id, duration=4400)
                                buffer_idx = replay_buffer.pre_add(state=state, action=action_idx, log_prob=log_prob,
                                                                   value=val, time_step=time)
                                recommended_vehicles[veh_id] = {'charging': False,
                                                                'station': charging_station_name,
                                                                'time': time,
                                                                'battery_level': battery_level,
                                                                'state': state,
                                                                'action_idx': action_idx,
                                                                'buffer_idx': buffer_idx,
                                                                'reward': 0,
                                                                'balance_reward': 0,
                                                                'log_prob': log_prob,
                                                                'val': val}
                                action_idx_value = int(action_idx)
                                if action_idx_value in small_indices:
                                    recommended_vehicles[veh_id]['balance_reward'] = 5
                                else:
                                    recommended_vehicles[veh_id]['balance_reward'] = 0
                            except:
                                pass
                    if veh_id in recommended_vehicles and not recommended_vehicles[veh_id]['charging']:
                        current_edge = traci.vehicle.getRoadID(veh_id)
                        charging_station_edge = charging_stations[recommended_vehicles[veh_id]['station']]
                        if current_edge == charging_station_edge:
                            print(f"Vehicle {veh_id} has arrived at charging station {recommended_vehicles[veh_id]['station']}.")
                            chargernum = charging_stations_details[recommended_vehicles[veh_id]['station']]['availableChargers']
                            parking_area_id = f"parkingArea{list(charging_stations.keys()).index(recommended_vehicles[veh_id]['station']) + 1}"
                            vnum = traci.parkingarea.getVehicleCount(parking_area_id)
                            vid = traci.parkingarea.getVehicleIDs(parking_area_id)
                            soc = []
                            for v_id in vid:
                                soc.append(float(traci.vehicle.getParameter(v_id, "device.battery.chargeLevel")))
                            soc.sort(reverse=True)
                            recommended_vehicles[veh_id]['reward'] = (battery_level - recommended_vehicles[veh_id]['battery_level']) / battery_capacity * 100
                            if len(soc) >= chargernum:
                                recommended_vehicles[veh_id]['wait_time'] = - ((max(battery_capacity * 0.8 - soc[vnum % chargernum], 0) + (vnum // chargernum - 1) * battery_capacity * uniform(0.6,0.7))
                                                                               / min((charging_stations_details[recommended_vehicles[veh_id]['station']]['chargingPower'] * 0.9 * 1000),100000) * 60)
                            else:
                                recommended_vehicles[veh_id]['wait_time'] = 0
                            if battery_level < 0.05 * battery_capacity:
                                recommended_vehicles[veh_id]['reward'] -= 10
                            recommended_vehicles[veh_id]['charging'] = True
                    if veh_id in recommended_vehicles and battery_level >= 0.8 * battery_capacity:
                        print(f"Vehicle {veh_id} has finished charging. New battery level: {battery_level}")
                        reward = - (time - recommended_vehicles[veh_id]['time']) / 60 + recommended_vehicles[veh_id]['wait_time']
                        succ_num += 1
                        replay_buffer.update_reward(recommended_vehicles[veh_id]['buffer_idx'],
                                                    recommended_vehicles[veh_id]['balance_reward'] + reward)
                        if traci.vehicle.isStopped(veh_id):
                            traci.vehicle.resume(veh_id)
                        del recommended_vehicles[veh_id]
                        episode_return += reward
            traci.close()
        if maxre < episode_return / (succ_num) and i > 20:
            maxre = episode_return / (succ_num)
            agent.save()
        return_list.append(episode_return / (succ_num))
        replay_buffer.finalize_trajectory()
        if succ_num > 0:
            agent.update(replay_buffer)
            agent.lr_change(i)
        df = pd.DataFrame({
            'Return': return_list,
            'Average_Critic_Loss': agent.critic_loss_list
        })
        df.to_excel('returns(MAPPO).xlsx', index=False)
    return return_list


num_agents = len(charging_stations_details)
agent = MAPPO(num_features, num_agents, actor_lr, critic_lr, gamma, eps_clip, k_epochs, lam)
return_list = train_mappo_agent(agent, num_episodes, replay_buffer)

agent.load()

costtime = []
drivingtime = []
waittingtime = []
chargingtime = []
jfi_avg = []
route_files = ["Scenario1.rou.xml", "Scenario2.rou.xml", "Scenario3.rou.xml", "Scenario4.rou.xml", "Scenario5.rou.xml", 'railway.rou.xml', 'railway2.rou.xml', 'railway3.rou.xml', 'railway4.rou.xml', 'railway5.rou.xml', 'railway600.rou.xml', 'railway720.rou.xml', 'railway.rou.xml', 'railway1200.rou.xml', 'railway1800.rou.xml']
last_route_files = "railway.rou.xml"

for a in range(15):
    print('场景', a + 1)
    episode_return = 0
    driving_time = 0
    waitting_time = 0
    charging_time = 0
    b_return = 0
    recommended_vehicles = {}
    succ_num = 0
    battery_level_fin = {}
    need_charge_vehicles = {}
    charging_actions = np.zeros(14)
    selected_route_file = route_files[a]
    with open("railway.sumocfg", "r") as f:
        config_content = f.read()
    new_config_content = config_content.replace(last_route_files, selected_route_file)
    with open("railway.sumocfg", "w") as f:
        f.write(new_config_content)
    print('last_route_files', last_route_files)
    print('selected_route_file', selected_route_file)
    last_route_files = selected_route_file
    traci.start(["E:/SUMO/bin/sumo.exe", "-c", "railway.sumocfg", "--start"])
    jfi_history = []
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep(traci.simulation.getTime() + 10)
        time = traci.simulation.getTime()

        # Compute station utilization from parked vehicles and charger capacity.
        utilization_rates = []
        for station, info in charging_stations_details.items():
            parking_area_id = f"parkingArea{list(charging_stations.keys()).index(station) + 1}"
            occupied_chargers = traci.parkingarea.getVehicleCount(parking_area_id)
            total_chargers = info['availableChargers']
            u_i = occupied_chargers / total_chargers
            utilization_rates.append(u_i)
        # Compute Jain's fairness index for the current time step.
        sum_u = sum(utilization_rates)
        sum_u_sq = sum([u ** 2 for u in utilization_rates])
        N = len(utilization_rates)
        if sum_u_sq > 0:
            jfi_t = (sum_u ** 2) / (N * sum_u_sq)
        else:
            jfi_t = 1.0  # Treat an entirely idle system as perfectly balanced.
        jfi_history.append(jfi_t)

        Graphupdate()
        for veh_id in traci.vehicle.getIDList():
            reward = 0
            battery_level = float(traci.vehicle.getParameter(veh_id, "device.battery.chargeLevel"))
            if veh_id not in recommended_vehicles:
                battery_level_threshold = 0.2
                if battery_level < battery_level_threshold * battery_capacity:
                    print(f"Vehicle {veh_id} needs charging. Current battery level: {battery_level}")
                    need_charge_vehicles[veh_id] = {'time': time, 'battery_level': battery_level}
                    pos = traci.vehicle.getPosition(veh_id)
                    x_normalized = pos[0] / 7100
                    y_normalized = pos[1] / 12000
                    angle = traci.vehicle.getAngle(veh_id) / 360
                    battery_level_normalized = battery_level / battery_capacity
                    current_edge = traci.vehicle.getRoadID(veh_id)
                    csi = []
                    for station, info in charging_stations_details.items():
                        parking_area_id = f"parkingArea{list(charging_stations.keys()).index(station) + 1}"
                        parknum = traci.parkingarea.getVehicleCount(parking_area_id)
                        availableChargingPoints = info['availableChargers'] - parknum
                        route = traci.simulation.findRoute(current_edge, charging_stations[station], routingMode=1)
                        csi.append(availableChargingPoints / 20)
                        csi.append(info['availableChargers'] / 20)
                        csi.append(info['chargingPower'] / 180)
                        csi.append(min(route.length / 16000, 1))
                        csi.append(min(route.travelTime / 3600, 1))
                    if recommended_vehicles == {}:
                        aver_action = np.zeros(14)
                        small_indices = np.array([])
                    else:
                        acs = [veh_info['action_idx'] for veh_info in recommended_vehicles.values()]
                        one_hot_vectors = []
                        for ac in acs:
                            one_hot = np.eye(14)[ac]
                            one_hot_vectors.append(one_hot)
                        one_hot_matrix = np.array(one_hot_vectors)
                        aver_action = np.mean(one_hot_matrix, axis=0)
                        small_indices = np.where(aver_action < 1 / 14)[0]
                    state = [Graph, torch.tensor([x_normalized, y_normalized, battery_level_normalized, angle], dtype=torch.float32),
                             torch.tensor(csi, dtype=torch.float32), torch.tensor(aver_action, dtype=torch.float32)]
                    action_idx, _, _ = agent.take_action(state,True)
                    charging_station_name = list(charging_stations.keys())[action_idx]
                    charging_station_edge = charging_stations[charging_station_name]
                    print(f"Recommending charging station {charging_station_name} to vehicle {veh_id}.")
                    route = traci.simulation.findRoute(current_edge, charging_station_edge, routingMode=1)
                    edge_list = route.edges
                    print(f"Edge list for vehicle {veh_id}: {edge_list}")
                    try:
                        traci.vehicle.setRoute(veh_id, edge_list)
                        parking_area_id = f"parkingArea{list(charging_stations.keys()).index(charging_station_name) + 1}"
                        traci.vehicle.setParkingAreaStop(veh_id, parking_area_id, duration=4400)
                        recommended_vehicles[veh_id] = {'charging': False,
                                                        'station': charging_station_name,
                                                        'time': time,
                                                        'battery_level': battery_level,
                                                        'state': state,
                                                        'action_idx': action_idx,
                                                        'reward': 0}
                    except:
                        pass
            if veh_id in recommended_vehicles:
                if not recommended_vehicles[veh_id]['charging']:
                    current_edge = traci.vehicle.getRoadID(veh_id)
                    charging_station_edge = charging_stations[recommended_vehicles[veh_id]['station']]
                    if current_edge == charging_station_edge:
                        print(f"Vehicle {veh_id} has arrived at charging station {recommended_vehicles[veh_id]['station']}.")
                        chargernum = charging_stations_details[recommended_vehicles[veh_id]['station']]['availableChargers']
                        parking_area_id = f"parkingArea{list(charging_stations.keys()).index(recommended_vehicles[veh_id]['station']) + 1}"
                        vnum = traci.parkingarea.getVehicleCount(parking_area_id)
                        vid = traci.parkingarea.getVehicleIDs(parking_area_id)
                        soc = []
                        for v_id in vid:
                            soc.append(float(traci.vehicle.getParameter(v_id, "device.battery.chargeLevel")))
                        soc.sort(reverse=True)
                        recommended_vehicles[veh_id]['reward'] = (battery_level - recommended_vehicles[veh_id]['battery_level']) / battery_capacity * 100
                        if len(soc) >= chargernum:
                            need_charge_vehicles[veh_id]['waitting_time'] = ((max(battery_capacity * 0.8 - soc[vnum % chargernum], 0) + (vnum // chargernum - 1) * battery_capacity * uniform(0.6,0.7))
                                                                               / min((charging_stations_details[recommended_vehicles[veh_id]['station']]['chargingPower'] * 0.9 * 1000),100000) * 60)
                        else:
                            need_charge_vehicles[veh_id]['waitting_time'] = 0
                        need_charge_vehicles[veh_id]['driving_time'] = (time - recommended_vehicles[veh_id]['time']) / 60
                        need_charge_vehicles[veh_id]['charging_time'] = (battery_capacity * 0.8 - battery_level) / min((charging_stations_details[recommended_vehicles[veh_id]['station']]['chargingPower'] * 0.9 * 1000),100000) * 60
                        if battery_level < 0.05 * battery_capacity:
                            recommended_vehicles[veh_id]['reward'] -= 10
                            battery_level_fin[veh_id] = battery_level / battery_capacity
                        recommended_vehicles[veh_id]['charging'] = True
                if battery_level >= 0.8 * battery_capacity:
                    print(f"Vehicle {veh_id} has finished charging. New battery level: {battery_level}")
                    waitting_time += need_charge_vehicles[veh_id]['waitting_time']
                    driving_time += need_charge_vehicles[veh_id]['driving_time']
                    charging_time += need_charge_vehicles[veh_id]['charging_time']
                    charging_actions[recommended_vehicles[veh_id]['action_idx']] += 1
                    reward = (time - recommended_vehicles[veh_id]['time']) / 60 + need_charge_vehicles[veh_id]['waitting_time']
                    b_return += recommended_vehicles[veh_id]['reward']
                    succ_num += 1
                    if traci.vehicle.isStopped(veh_id):
                        traci.vehicle.resume(veh_id)
                    del recommended_vehicles[veh_id]
            episode_return += reward
    print('episode_return:', episode_return / succ_num)
    print('driving_time:', driving_time / succ_num)
    print('waitting_time:', waitting_time / succ_num)
    print('charging_time:', charging_time / succ_num)
    print('b_return:', -b_return / succ_num)
    print('battery_level_fin', battery_level_fin)
    print('recommended_vehicles', len(recommended_vehicles))
    print('succ_num:', succ_num)
    print('need_charge_vehicles', len(need_charge_vehicles))
    print('charging_actions', charging_actions)
    traci.close()
    costtime.append(episode_return / succ_num)
    drivingtime.append(driving_time / succ_num)
    waittingtime.append(waitting_time / succ_num)
    chargingtime.append(charging_time / succ_num)
    jfi_avg.append(sum(jfi_history) / len(jfi_history))
    df = pd.DataFrame({'costtime': costtime, 'drivingtime': drivingtime, 'waittingtime': waittingtime,
                       'chargingtime': chargingtime, 'jfi_avg': jfi_avg})
    df.to_excel('costtime(MAPPO+GCN+A).xlsx', index=False)
