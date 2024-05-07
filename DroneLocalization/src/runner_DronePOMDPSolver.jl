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
using CairoMakie
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
#### THIS METHOD WILL NOT WORK RUN OUT OF MEMORY ####
# sarsop_solver = SARSOPSolver(max_iterations=100, tolerance=1e-6, verbose=false)
# sarsop_solution = solve(sarsop_solver, m)
# println("Solved.")
# sarsop_rolled = [simulate(RolloutSimulator(max_steps=200), m, sarsop_solution, up) for _ in 1:300]
# println("Rolled.")
# println("Mean:", mean(sarsop_rolled))
# println("STD:", std(sarsop_rolled))
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

# # Run the simulation - SARSOP
# history_SARSOP = stepthrough(m, sarsop_solution, up, "s,a,r,sp", max_steps=1000)
# # Record the trajectory and rewards - SARSOP
# for step in history_SARSOP
#     push!(reward_trajectory_SARSOP, step.r)
# end
# println("SARSOP Simulation completed and trajectory saved")

# Run the simulation - POMCPOW
history_POMCPOW = stepthrough(m, pomcpow_p, up, "s,a,r,sp", max_steps=1000)
# Record the trajectory and rewards - POMCPOW
for step in history_POMCPOW
    push!(reward_trajectory_POMCPOW, step.r)
end
println("POMCPOW Simulation completed and trajectory saved")


# ## Plotting the trajectory
QMDP_reward_plot = Plots.plot(reward_trajectory_QMDP, xlabel = "Step #", ylabel = "Reward", title = "Reward Trajectory", color = :red, label = "QMDP")
Plots.plot!(reward_trajectory_POMCPOW, color = :blue, label = "POMCPOW")

for ii in 1:10
    # Run the simulation - QMDP
    history_QMDP = stepthrough(m, QMDP_SOLUTION, up, "s,a,r,sp", max_steps=1000)
    # Record the trajectory and rewards - QMDP
    for step in history_QMDP
        push!(reward_trajectory_QMDP, step.r)
    end
    println("QMDP Simulation completed and trajectory saved")

    # # Run the simulation - SARSOP
    # history_SARSOP = stepthrough(m, sarsop_solution, up, "s,a,r,sp", max_steps=1000)
    # # Record the trajectory and rewards - SARSOP
    # for step in history_SARSOP
    #     push!(reward_trajectory_SARSOP, step.r)
    # end
    # println("SARSOP Simulation completed and trajectory saved")

    # Run the simulation - POMCPOW
    history_POMCPOW = stepthrough(m, pomcpow_p, up, "s,a,r,sp", max_steps=1000)
    # Record the trajectory and rewards - POMCPOW
    for step in history_POMCPOW
        push!(reward_trajectory_POMCPOW, step.r)
    end
    println("POMCPOW Simulation completed and trajectory saved")


    # ## Plotting the trajectory
    Plots.plot!(reward_trajectory_QMDP, color = :red, label = false)
    Plots.plot!(reward_trajectory_POMCPOW, color = :blue, label = false)
end

############################# Result Plotting #############################
#Rollout plots
# Create three plots
qmdp_rolled_plot = Plots.plot(qmdp_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory - QMDP", label = false)
# sarsop_rolled_plot = plot(sarsop_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory")
pomcpow_rolled_plot = Plots.plot(pomcpow_rolled, xlabel = "Rollout Iteration #", ylabel = "Reward", title = "Rollout Trajectory - POMCPOW", color = :red, label = false)

# Combine these plots into a 1x3 layout
# P_Rolled = Plots.plot(qmdp_rolled_plot,sarsop_rolled_plot,pomcpow_rolled_plot, layout = (1, 3))


#Reward Trajectory Plots - Combined
# P_Reward = Plots.plot(reward_trajectory_QMDP,reward_trajectory_SARSOP,reward_trajectory_POMCPOW,layout = (1, 3))



## Trying to get the alpha vectors in 3D
##Get Alpha Vectors
# qmdp_alphas = QMDP_SOLUTION.alphas
# sarsop_alphas = sarsop_solution.alphas
## Normalize the alpha vectors
# sarsop_alphas_normalized = sarsop_alphas ./ sum(sarsop_alphas, dims=2)

# Create the ternary plot
# ternaryplot(title = "Three-Dimensional Drone POMDP Alpha Vectors", size = (800, 800))
# ternaryscatter!(qmdp_alphas_normalized, label="QMDP", color=:blue)
# ternaryscatter!(sarsop_alphas_normalized, label="SARSOP", color=:red)
# p2 = plot(QMDP_SOLUTION.alphas,xlabel = "Belief", ylabel = "Value", title = "QMDP Solution Alpha Vectors", label=["α_1" "α_2" "α_3"]) ## TODO Fix this bruddah




function MakieRender(m::DronePOMDP, step)
    ## Preallocation ##
    POSObstacles = zeros(length(m.obstacles), 3)
    indexCounter = 1;
    POSDrone = zeros(1, 3)
    POSTarget = zeros(1, 3)
    POSBystander = zeros(1, 3)

    nx, ny, nz = m.size

    ## Combing through the Step Data ##
    for x in 1:nx, y in 1:ny, z in 1:nz
        cell = SVector(x, y, z)
        if cell in m.obstacles
            POSObstacles[indexCounter, 1] = x
            POSObstacles[indexCounter, 2] = y
            POSObstacles[indexCounter, 3] = z
            indexCounter = indexCounter + 1
        end
        if cell == step[:sp].drone
            POSDrone[1] = x
            POSDrone[2] = y
            POSDrone[3] = z
        end
        if cell == step[:sp].target
            POSTarget[1] = x
            POSTarget[2] = y
            POSTarget[3] = z
        end
        if cell == step[:sp].bystander
            POSBystander[1] = x
            POSBystander[2] = y
            POSBystander[3] = z
        end
            
    end

    ## Plotting ##
    function mesh_cube(POSITION, size=1, color=:darkgrey)
        s = size / 2
        x, y, z = POSITION
        verts = [
            x - s y - s z - s;
            x + s y - s z - s;
            x + s y + s z - s;
            x - s y + s z - s;
            x - s y - s z + s;
            x + s y - s z + s;
            x + s y + s z + s;
            x - s y + s z + s;
        ]
        faces = [
            1 2 3;
            1 3 4;
            5 6 7;
            5 7 8;
            1 2 6;
            1 6 5;
            2 3 7;
            2 7 6;
            3 4 8;
            3 8 7;
            4 1 5;
            4 5 8;
        ]
        # vertices = hcat(verts...)
        mesh!(verts, faces, color = color, shading = FastShading)
    end


    function mesh_ground(POSITION, size=1, color = :red)
        s = size / 2
        x, y, z = POSITION
        # Define the vertices of the pyramid
        verts = [
            x - s y - s z - s;  # Bottom left corner
            x + s y - s z - s;  # Bottom right corner
            x + s y + s z - s;  # Top right corner
            x - s y + s z - s;  # Top left corner
            x     y     z + s;  # Apex
        ]
        # Define the faces of the pyramid
        faces = [
            1 2 5;  # First triangle (front)
            2 3 5;  # Second triangle (right)
            3 4 5;  # Third triangle (back)
            4 1 5;  # Fourth triangle (left)
        ]

        mesh!(
            verts,
            faces,
            color = color,
            shading = FastShading
        )
    end

    #Initialize render
    # scene = meshscatter(0,0,0, color = :blue, markersize = 0.05)

    # Plotting Drone
    scene = meshscatter(POSDrone[1],
    POSDrone[2],
    POSDrone[3],
    color = :green,
    markersize = 0.5)
    

    # Plotting Obstacles
    for i in 1:size(POSObstacles, 1)
        mesh_cube(POSObstacles[i, :])
    end

    # Plotting Target
    mesh_ground(POSTarget, 1, :red)
    # meshscatter!(POSTarget[1],
    # POSTarget[2],
    # POSTarget[3],
    # color = :orange,
    # markersize = 0.4)

    # Plotting Bystander
    mesh_ground(POSBystander, 1, :blue)
    # meshscatter!(POSBystander[1],
    # POSBystander[2],
    # POSBystander[3],
    # color = :purple,
    # markersize = 0.4)

    # Plotting Measurement Lines
    if step[:a] == :measure
        o = step[:o]
        # (POSDrone[1], POSDrone[2], POSDrone[3]) #
        forward  = cat(POSDrone[1:3], [POSDrone[1] + o[1] + .5,     POSDrone[2],            POSDrone[3]],        dims=(2,2))
        backward = cat(POSDrone[1:3], [POSDrone[1] - o[2] - .5,     POSDrone[2],            POSDrone[3]],        dims=(2,2))
        left     = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2] - o[3] - .5,     POSDrone[3]],        dims=(2,2))
        right    = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2] + o[4] + .5,     POSDrone[3]],        dims=(2,2))
        up       = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2],            POSDrone[3] - o[5] - .5], dims=(2,2))
        down     = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2],            POSDrone[3] + o[6] + .5], dims=(2,2))

        # Plotting
        GLMakie.lines!(forward,   linestyle = :dash, color = :red)
        GLMakie.lines!(backward,  linestyle = :dash, color = :red)
        GLMakie.lines!(left,      linestyle = :dash, color = :red)
        GLMakie.lines!(right,     linestyle = :dash, color = :red)
        GLMakie.lines!(up,        linestyle = :dash, color = :red)
        GLMakie.lines!(down,      linestyle = :dash, color = :red)
    end

    # ## Display ##
    scene
end
for step in stepthrough(m, policy, up, max_steps=10)
    push!(history, step)
end

MakieRender(m, history[1])
