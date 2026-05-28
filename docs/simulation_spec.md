# Simulation Specification

## 1. Scope

This project simulates DQN-based dynamic RAN slicing for an O-RAN-enabled IoV scenario.

The simulation scope is limited to:
- Single-cell RAN slicing
- One gNB
- Two slices:
  - Ambulance Slice
  - Ordinary Traffic Slice
- DQN agent logically deployed at the Near-RT RIC
- DQN controls only inter-slice PRB allocation

The simulation does not include:
- End-to-end slicing
- Core network slicing
- Transport network slicing
- Multi-gNB association
- Handover
- Power allocation
- VNF placement
- Real E2 interface
- Real O-RAN xApp deployment

## 2. Units

- Time: second
- Decision interval: 1 second
- Traffic and queue: Mbit
- Capacity and throughput: Mbit/s
- Latency: second
- PRB: integer number of physical resource blocks

## 3. System Parameters

- Total PRB: 100
- Number of slices: 2
- Episode length: 200 steps
- Maximum number of ambulance vehicles: 3
- Number of ordinary vehicles: 20 to 70
- Number of eMBB users: 5 to 25
- Maximum number of ordinary/eMBB users for normalization: 100

## 4. Traffic Model

Ambulance Slice traffic follows a Markov-modulated Poisson model.

Ambulance traffic states:
- normal
- emergency

Transition probabilities:
- p_on = 0.03
- p_off = 0.08

Ambulance packet size:
- P_A = 0.01 Mbit

Ambulance packet arrival rates:
- lambda_A_normal = 5 packets/s/ambulance
- lambda_A_emergency = 50 packets/s/ambulance

Ordinary Traffic Slice:
- A_O(t) = A_V(t) + A_E(t)

Ordinary vehicle packet size:
- P_V = 0.02 Mbit

Ordinary vehicle arrival rate:
- lambda_V = 2 packets/s/user

eMBB packet size:
- P_E = 0.1 Mbit

eMBB arrival rates:
- lambda_E_normal = 5 packets/s/user
- lambda_E_surge = 8 packets/s/user

## 5. Channel and Capacity Model

The simulation uses UE-level spectral efficiency.

For each UE i:
- eta_i(t) is the spectral efficiency.

Channel states:
- poor: eta in [0.5, 1.5]
- normal: eta in [1.5, 3.0]
- good: eta in [3.0, 5.0]

Channel state probabilities:
- poor: 0.15
- normal: 0.65
- good: 0.20

Base PRB rate:
- R_PRB = 0.1 Mbit/s/PRB when eta = 1

UE capacity:
- C_i(t) = b_i(t) * R_PRB * eta_i(t)

Slice capacity:
- C_s(t) = sum C_i(t) for all UEs in slice s

## 6. Queue and Latency Model

Queue update:
- Q_s(t+1) = max(0, Q_s(t) + A_s(t) - C_s(t) * delta_t)

Since delta_t = 1 second:
- Q_s(t+1) = max(0, Q_s(t) + A_s(t) - C_s(t))

Queue limits:
- Q_A_max = 20 Mbit
- Q_O_max = 100 Mbit

Queue values are clipped at their maximum values.

Ambulance estimated RAN-side latency:
- L_A(t) = (Q_A(t) + A_A(t)) / (C_A(t) + epsilon)

epsilon:
- 1e-6

Ambulance SLA threshold:
- L_A_max = 0.1 second

Ordinary throughput target:
- R_O_target = 8 Mbit/s

## 7. State Space

The state vector is:

s_t = [
  n_A,
  n_O,
  rho_A,
  rho_O,
  q_A,
  q_O,
  l_A,
  r_O,
  u,
  eta_A_avg,
  eta_O_avg,
  alpha_A_prev
]

Definitions:
- n_A = N_A / N_A_max
- n_O = N_O / N_O_max
- rho_A = A_A / A_A_max
- rho_O = A_O / A_O_max
- q_A = Q_A / Q_A_max
- q_O = Q_O / Q_O_max
- l_A = L_A / L_A_max, clipped to [0, 1]
- r_O = R_O / R_O_target, clipped to [0, 1]
- u = used_PRB / total_PRB
- eta_A_avg = average spectral efficiency of Ambulance Slice / eta_max
- eta_O_avg = average spectral efficiency of Ordinary Traffic Slice / eta_max
- alpha_A_prev = previous PRB ratio for Ambulance Slice

All state values fed into the DQN must be clipped to [0, 1].

## 8. Action Space

The DQN action space has 8 discrete actions.

Action mapping:
- 0: Ambulance 10%, Ordinary 90%
- 1: Ambulance 20%, Ordinary 80%
- 2: Ambulance 30%, Ordinary 70%
- 3: Ambulance 40%, Ordinary 60%
- 4: Ambulance 50%, Ordinary 50%
- 5: Ambulance 60%, Ordinary 40%
- 6: Ambulance 70%, Ordinary 30%
- 7: Ambulance 80%, Ordinary 20%

The DQN controls only inter-slice PRB allocation.
The RAN scheduler handles intra-slice allocation.

## 9. Reward Function

Reward:

r_t = -w1 * E_L(t)
      -w2 * V_A(t)
      +w3 * T_O(t)
      -w4 * W_A(t)
      -w5 * abs(alpha_A(t) - alpha_A(t-1))

Where:
- E_L(t) = max(0, L_A(t) / L_A_max - 1)
- V_A(t) = 1 if L_A(t) > L_A_max, else 0
- T_O(t) = min(1, R_O(t) / R_O_target)
- W_A(t) = alpha_A(t) if rho_A(t) < 0.2 and alpha_A(t) > 0.5, else 0

Reward weights:
- w1 = 3.0
- w2 = 10.0
- w3 = 1.0
- w4 = 0.5
- w5 = 0.2

## 10. Baselines

The following baselines must be implemented:

1. Static slicing:
   - alpha_A = 0.3

2. Priority-based slicing:
   - if N_A > 0: alpha_A = 0.7
   - else: alpha_A = 0.1

3. Load-based slicing:
   - alpha_A = rho_A / (rho_A + rho_O + epsilon)
   - quantize alpha_A to the nearest valid action

4. Greedy SLA-based slicing:
   - if L_A > L_A_max: increase alpha_A by 0.1, max 0.8
   - if L_A < 0.7 * L_A_max and R_O < R_O_target: decrease alpha_A by 0.1, min 0.1

5. Random slicing:
   - choose a random action
   - used only for sanity check

## 11. Metrics

The simulation must log:
- average ambulance latency
- 95th percentile ambulance latency
- ambulance SLA violation rate
- ambulance QoS satisfaction rate
- ordinary average throughput
- ordinary throughput deficit rate
- average PRB utilization
- episode reward
- action distribution
- reward components

## 12. DQN Training Setup

Initial DQN settings:
- policy: MlpPolicy
- input dimension: 12
- output dimension: 8
- hidden layers: [128, 128]
- activation: ReLU
- learning rate: 5e-4
- gamma: 0.99
- batch size: 64
- replay buffer size: 50000
- learning starts: 1000
- target update interval: 1000
- exploration initial epsilon: 1.0
- exploration final epsilon: 0.05
- exploration fraction: 0.3
- initial total timesteps: 100000