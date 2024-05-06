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
using FileIO
using GeometryBasics: coordinates
using Meshes: Scale, Translate


# Define the POMDP Environment
m = DronePOMDP()
println("Environment Created")
up = DiscreteUpdater(m)
println("Updater Created")
policy = FunctionPolicy(o->:measure)
println("Policy Created")
history = []

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

MakieRender(m, first(history))
