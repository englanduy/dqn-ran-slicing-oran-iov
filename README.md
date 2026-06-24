# DQN-based Dynamic RAN Slicing in O-RAN for Emergency Vehicles in IoV

This repository implements a Python simulation for DQN-based dynamic RAN slicing.
The system models a single-cell RAN with two slices:

- Ambulance Slice
- Ordinary Traffic Slice, serving ordinary vehicles only

The DQN agent is logically deployed at the Near-RT RIC and controls inter-slice PRB allocation.

The Ordinary Traffic Slice has no eMBB users, surge process, or additional
high-rate load. Its per-slot traffic is
`N_O_pkt(t) ~ Poisson(lambda_O * n_O(t) * delta_t)` and
`A_O(t) = P_O * N_O_pkt(t)`.

Existing models and result artifacts produced before this change were generated
with eMBB traffic. Keep them as historical artifacts, but do not use them as
pretrained models or final results for the ordinary-vehicle-only environment.
