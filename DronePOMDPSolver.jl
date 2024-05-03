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
import .DroneLocalization: DronePOMDP, DroneState

######################
# Drone Localization Solution With POMDP Methods
######################

# Define the POMDP Environment
m = DronePOMDP()

function pomcpow_solve(m)
    solver = POMCPOWSolver(criterion=MaxUCB(15.0), 
                           tree_queries=100, 
                           estimate_value=FORollout(SparseValueIterationSolver(max_iterations = 1500)),
                           default_action=first(actions(m)))
    return solve(solver, m)
end

pomcpow_p = pomcpow_solve(m)
println("Solved.")

up = DiscreteUpdater(m)

pomcpow_rolled = [simulate(RolloutSimulator(), m, pomcpow_p, up) for _ in 1:100]
println("Rolled.")
println("Mean:", mean(pomcpow_rolled))
println("STD:", std(pomcpow_rolled))

using POMDPGifs
import Cairo, Fontconfig

pomcpow_rolled_plot = plot(pomcpow_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")

#makegif(m, pomcpow_p, up, filename="localization.gif")

#@show mean(simulate(RolloutSimulator(), pomdp, pomcpow_p, up) for _ in 1:10) 

