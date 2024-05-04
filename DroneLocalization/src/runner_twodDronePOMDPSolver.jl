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
include("DroneLocalization.jl")
# include("twoDDroneLocalization.jl")
import .DroneLocalization: DronePOMDP, DroneState
# import .twoDDroneLocalization: twoDDronePOMDP, twoDDroneState

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
# m2 = twoDDronePOMDP()
println("Environment Created")
## Define the updater ##
# Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
up = DiscreteUpdater(m)
# up2 = DiscreteUpdater(m2)
println("Updater Defined")
############################# QMDP Solver #############################
# qmdp_p = qmdp_solve(m)
# println("Solved")
QMDP_solver = QMDPSolver(max_iterations=100, belres=1e-6, verbose=false)
QMDP_SOLUTION = solve(QMDP_solver, m)

# twoDQMDP_SOLUTION = solve(QMDP_solver, m2)
println("Solved")

qq = 500
# qq = 1
qmdp_rolled = [simulate(RolloutSimulator(max_steps=1000), m, QMDP_SOLUTION, up) for _ in 1:qq]
println("Rolled")

println("Mean:", mean(qmdp_rolled))
println("STD:", std(qmdp_rolled))
############################# Trajectory Saver #############################


# Initialize lists to store the trajectory and rewards
state_trajectory = []
action_trajectory = []
reward_trajectory = []

# Run the simulation
history = stepthrough(m, QMDP_SOLUTION, up, "s,a,r,sp", max_steps=1000)

# Record the trajectory and rewards
for step in history
    push!(state_trajectory, step.s)
    push!(action_trajectory, step.a)
    push!(reward_trajectory, step.r)
end

println("Simulation completed and trajectory saved")
# println("State Trajectory: ", state_trajectory)
# println("Action Trajectory: ", action_trajectory)
# println("Reward Trajectory: ", reward_trajectory)

# ## Plotting the trajectory
reward_plot = plot(reward_trajectory, xlabel = "Step #", ylabel = "Reward", title = "Reward Trajectory")
@show(reward_plot)
############################# SARSOP Solver #############################
sarsop_solver = SARSOPSolver(max_iterations=100, tolerance=1e-6, verbose=false)
sarsop_solution = solve(sarsop_solver, m)
println("Solved.")
sarsop_rolled = [simulate(RolloutSimulator(max_steps=100), m, sarsop_solution, up) for _ in 1:100]
println("Rolled.")
println("Mean:", mean(sarsop_rolled))
println("STD:", std(sarsop_rolled))
############################# POMCPOW Solution #############################
println("POMCPOW Solver")
function pomcpow_solve(m)
    solver = POMCPOWSolver(tree_queries=200, 
                            criterion = MaxUCB(30.0), 
                            default_action=last(actions(m)), 
                            estimate_value=FORollout(SparseValueIterationSolver(max_iterations = 1500)))
    return solve(solver, m)
end
pomcpow_p = pomcpow_solve(m)
println("Solved.")
pomcpow_rolled = [simulate(RolloutSimulator(max_steps=100), m, pomcpow_p, up) for _ in 1:100]
println("Rolled.")
println("Mean:", mean(pomcpow_rolled))
println("STD:", std(pomcpow_rolled))
############################# Result Plotting #############################
p2 = plot(QMDP_SOLUTION.alphas,xlabel = "Belief", ylabel = "Value", title = "QMDP Solution Alpha Vectors", label=["α_1" "α_2" "α_3"]) ## TODO Fix this bruddah

##Rollout plots
# Create three plots
qmdp_rolled_plot = plot(qmdp_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")
sarsop_rolled_plot = plot(sarsop_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")
pomcpow_rolled_plot = plot(pomcpow_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")

# Combine these plots into a 1x3 layout
plot(qmdp_rolled_plot,sarsop_rolled_plot,pomcpow_rolled_plot, layout = (1, 3))

# ----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
using POMDPGifs
import Cairo, Fontconfig # needed to display properly
## TODO - See what happenes when max_steps is increased
makegif(m, QMDP_SOLUTION, up, max_steps=50, filename="localization.gif")

# # You can render a single frame like this
# using POMDPTools: stepthrough, render
# using Compose: draw, PNG

# history = []
# for step in stepthrough(m, pomcpow_p, up, max_steps=10)
#     push!(history, step)
# end
# displayable_object = render(m, last(history))
# # display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
# draw(PNG("localization.png"), displayable_object)