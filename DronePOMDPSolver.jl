using POMDPs
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using QuickPOMDPs: QuickPOMDP
using POMDPModels
using NativeSARSOP: SARSOPSolver
using POMDPTesting: has_consistent_distributions
using QMDP
using LinearAlgebra: dot
using Plots
using Statistics: mean, std
using POMCPOW
using DiscreteValueIteration
include("DroneLocalization.jl")
import .DroneLocalization: DronePOMDP, DroneState, DroneAction, DroneObservation, DroneTransition, DroneReward, DroneDistribution


######################
# Drone Localization Solution With POMDP Methods
######################

# Define the POMDP Environment
m = DronePOMDP()