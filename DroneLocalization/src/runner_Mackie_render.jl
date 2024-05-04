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
using GLMakie
using StaticArrays
include("DroneLocalization.jl")
import .DroneLocalization: DronePOMDP, DroneState
import Cairo

# Define the POMDP Environment
m = DronePOMDP()
println("Environment Created")
up = DiscreteUpdater(m)
println("Updater Created")
policy = FunctionPolicy(o->:measure)
println("Policy Created")
history = []

function MakeRender(m::DronePOMDP, step)
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
    # Plotting Obstacles
    scene = meshscatter(POSObstacles[:, 1],
    POSObstacles[:, 2],
    POSObstacles[:, 3],
    color = :darkgrey,
    markersize = 0.3)

    # Plotting Drone
    meshscatter!(POSDrone[1],
    POSDrone[2],
    POSDrone[3],
    color = :green,
    markersize = 0.5)

    # Plotting Target
    meshscatter!(POSTarget[1],
    POSTarget[2],
    POSTarget[3],
    color = :orange,
    markersize = 0.4)

    # Plotting Bystander
    meshscatter!(POSBystander[1],
    POSBystander[2],
    POSBystander[3],
    color = :purple,
    markersize = 0.4)

    # Plotting Measurement Lines
    if step[:a] == :measure
        o = step[:o]
        # (POSDrone[1], POSDrone[2], POSDrone[3]) #
        forward  = cat(POSDrone[1:3], [POSDrone[1] + o[1],     POSDrone[2],            POSDrone[3]],        dims=(2,2))
        backward = cat(POSDrone[1:3], [POSDrone[1] - o[2],     POSDrone[2],            POSDrone[3]],        dims=(2,2))
        left     = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2] - o[3],     POSDrone[3]],        dims=(2,2))
        right    = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2] + o[4],     POSDrone[3]],        dims=(2,2))
        up       = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2],            POSDrone[3] + o[5]], dims=(2,2))
        down     = cat(POSDrone[1:3], [POSDrone[1],            POSDrone[2],            POSDrone[3] - o[6]], dims=(2,2))

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

MakeRender(m, last(history))
