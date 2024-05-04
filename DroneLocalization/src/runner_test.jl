using POMDPs
using POMDPTools: transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using POMDPTools: stepthrough, render
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
import .DroneLocalization: DronePOMDP, DroneState

# Define the POMDP Environment
m = DronePOMDP()
println("Environment Created")
up = DiscreteUpdater(m)
println("Updater Created")
policy = FunctionPolicy(o->:measure)
println("Policy Created")
history = []

for step in stepthrough(m, policy, up, max_steps=10)
    push!(history, step)
end

render(m, last(history))