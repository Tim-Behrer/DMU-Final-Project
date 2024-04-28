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
@show("Environment Created")
## Define the updater ##
# Z(m::POMDP, a, sp, o) = pdf(observation(m, a, sp), o)
# T(m::POMDP, s, a, sp) = pdf(transition(m, s, a), sp)
up = DiscreteUpdater(m)
@show("Updater Defined")
############################# QMDP Solver #############################
# qmdp_p = qmdp_solve(m)
# @shwo("Solved")
QMDP_solver = QMDPSolver(max_iterations=100, belres=1e-6, verbose=false)
QMDP_SOLUTION = solve(QMDP_solver, m)
@show("Solved")

qmdp_rolled = [simulate(RolloutSimulator(max_steps=100), m, qmdp_p, up) for _ in 1:500]
@show("Rolled")
############################# SARSOP Solver #############################


############################# Result Plotting #############################
p2 = plot(QMDP_SOLUTION.alphas,xlabel = "Belief", ylabel = "Value", title = "QMDP Solution Alpha Vectors", label=["α_1" "α_2" "α_3"])

# ----------------
# Visualization
# (all code below is optional)
#----------------

# You can make a gif showing what's going on like this:
using POMDPGifs
import Cairo, Fontconfig # needed to display properly

makegif(m, qmdp_p, up, max_steps=30, filename="localization.gif")

# You can render a single frame like this
using POMDPTools: stepthrough, render
using Compose: draw, PNG

history = []
for step in stepthrough(m, pomcpow_p, up, max_steps=10)
    push!(history, step)
end
displayable_object = render(m, last(history))
# display(displayable_object) # <-this will work in a jupyter notebook or if you have vs code or ElectronDisplay
draw(PNG("localization.png"), displayable_object)
