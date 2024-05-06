using POMDPs
using POMDPTools: stepthrough, transition_matrices, reward_vectors, SparseCat, Deterministic, RolloutSimulator, DiscreteBelief, FunctionPolicy, ordered_states, ordered_actions, DiscreteUpdater
using POMDPSimulators
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
using TernaryPlots
using Colors
include("DroneLocalization.jl")
import .DroneLocalization: DronePOMDP, DroneState

######################
# Drone Localization Solution With POMDP Methods
######################

######################
## Updater
######################
struct DroneUpdater{M<:POMDP} <: Updater
    m::M
end

######################
## Policy Solvers
######################
struct DroneAlphaVectorPolicy{A} <: Policy
    alphas::Vector{Vector{Float64}}
    alpha_actions::Vector{A}
end

function POMDPs.action(p::DroneAlphaVectorPolicy, b::DiscreteBelief)

    # Fill in code to choose action based on alpha vectors
    belief = beliefvec(b)
    max_value = -Inf
    max_action = first(actions(b.pomdp))

    values = [dot(alpha, belief) for alpha in p.alphas]

    return p.alpha_actions[argmax(values)]
end

beliefvec(b::DiscreteBelief) = b.b # this function may be helpful to get the belief as a vector in stateindex order

######################
## QMDP solver
######################
function qmdp_solve(m,discount = discount(m))

    function qmdp_valueIteration(m,discount, ϵ = 1e-3)
        V = rand(length(states(m)))
        T = transition_matrices(m)
        R = reward_vectors(m)
        num_states = length(states(m))
        Δ = Inf


        while Δ > ϵ
            V_prime = copy(V)
            for s in 1:num_states
                V_prime[s] = maximum(R[a][s] + discount * sum(T[a][s, sp] * V[sp] for sp in 1:num_states) for a in actions(m))
            end
            Δ = maximum(abs.(V_prime - V))
            V = V_prime
        end
        return V
    end

    V = qmdp_valueIteration(m,discount)
    acts = actiontype(m)[]
    alphas = Vector{Float64}[]

    for a in actions(m)
        # Fill in alpha vector calculation
        # Note that the ordering of the entries in the alpha vectors must be consistent with stateindex(m, s) (states(m) does not necessarily obey this order, but ordered_states(m) does.)
        alpha = zeros(length(states(m)))
        for s in states(m)
            alpha[stateindex(m, s)] = reward(m,s,a) + discount * sum(T(m,s,a,sp) * V[stateindex(m,sp)] for sp in ordered_states(m))
        end
        push!(alphas, alpha)
        push!(acts, a)
    end

    return DroneAlphaVectorPolicy(alphas, acts)
end

# Define the POMDP Environment
m = DronePOMDP()
println("Environment Created")

up = DiscreteUpdater(m)
println("Updater Defined")
############################# QMDP Solver #############################

QMDP_solver = QMDPSolver(max_iterations=100, belres=1e-6, verbose=false)
QMDP_SOLUTION = solve(QMDP_solver, m)

# twoDQMDP_SOLUTION = solve(QMDP_solver, m2)
println("Solved")

qmdp_rolled = [simulate(RolloutSimulator(max_steps=200), m, QMDP_SOLUTION, up) for _ in 1:300]
println("Rolled")
println("Mean:", mean(qmdp_rolled))
println("STD:", std(qmdp_rolled))

############################# SARSOP Solver #############################
sarsop_solver = SARSOPSolver(max_iterations=100, tolerance=1e-6, verbose=false)
sarsop_solution = solve(sarsop_solver, m)
println("Solved.")
sarsop_rolled = [simulate(RolloutSimulator(max_steps=200), m, sarsop_solution, up) for _ in 1:300]
println("Rolled.")
println("Mean:", mean(sarsop_rolled))
println("STD:", std(sarsop_rolled))
############################# POMCPOW Solution #############################
println("POMCPOW Solver")
function pomcpow_solve(m)
    solver = POMCPOWSolver(tree_queries=200, 
                            criterion = MaxUCB(15.0), 
                            default_action=first(actions(m)), 
                            estimate_value=FORollout(SparseValueIterationSolver(max_iterations = 1500)))
    return solve(solver, m)
end
pomcpow_p = pomcpow_solve(m)
println("Solved.")
pomcpow_rolled = [simulate(RolloutSimulator(max_steps=100), m, pomcpow_p, up) for _ in 1:300]
println("Rolled.")
println("Mean:", mean(pomcpow_rolled))
println("STD:", std(pomcpow_rolled))
############################# Trajectory Saver #############################


# Initialize lists to store the trajectory and rewards
reward_trajectory_QMDP = []
reward_trajectory_SARSOP = []
reward_trajectory_POMCPOW = []

# Run the simulation - QMDP
history_QMDP = stepthrough(m, QMDP_SOLUTION, up, "s,a,r,sp", max_steps=1000)
# Record the trajectory and rewards - QMDP
for step in history_QMDP
    push!(reward_trajectory_QMDP, step.r)
end
println("QMDP Simulation completed and trajectory saved")

# Run the simulation - SARSOP
history_SARSOP = stepthrough(m, sarsop_solution, up, "s,a,r,sp", max_steps=1000)
# Record the trajectory and rewards - SARSOP
for step in history_SARSOP
    push!(reward_trajectory_SARSOP, step.r)
end
println("SARSOP Simulation completed and trajectory saved")

# Run the simulation - POMCPOW
history_POMCPOW = stepthrough(m, pomcpow_p, up, "s,a,r,sp", max_steps=1000)
# Record the trajectory and rewards - POMCPOW
for step in history_POMCPOW
    push!(reward_trajectory_POMCPOW, step.r)
end
println("POMCPOW Simulation completed and trajectory saved")


# ## Plotting the trajectory
QMDP_reward_plot = plot(reward_trajectory, xlabel = "Step #", ylabel = "Reward", title = "Reward Trajectory")

############################# Result Plotting #############################
#Rollout plots
# Create three plots
qmdp_rolled_plot = plot(qmdp_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")
sarsop_rolled_plot = plot(sarsop_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")
pomcpow_rolled_plot = plot(pomcpow_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")

# Combine these plots into a 1x3 layout
P_Rolled = plot(qmdp_rolled_plot,sarsop_rolled_plot,pomcpow_rolled_plot, layout = (1, 3))


#Reward Trajectory Plots - Combined
P_Reward = plot(reward_trajectory_QMDP,reward_trajectory_SARSOP,reward_trajectory_POMCPOW,layout = (1, 3))



## Trying to get the alpha vectors in 3D
##Get Alpha Vectors
qmdp_alphas = QMDP_SOLUTION.alphas
# sarsop_alphas = sarsop_solution.alphas
## Normalize the alpha vectors
# sarsop_alphas_normalized = sarsop_alphas ./ sum(sarsop_alphas, dims=2)

# Create the ternary plot
# ternaryplot(title = "Three-Dimensional Drone POMDP Alpha Vectors", size = (800, 800))
# ternaryscatter!(qmdp_alphas_normalized, label="QMDP", color=:blue)
# ternaryscatter!(sarsop_alphas_normalized, label="SARSOP", color=:red)
# p2 = plot(QMDP_SOLUTION.alphas,xlabel = "Belief", ylabel = "Value", title = "QMDP Solution Alpha Vectors", label=["α_1" "α_2" "α_3"]) ## TODO Fix this bruddah
