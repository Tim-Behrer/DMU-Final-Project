## Reference: HW6.jl 
## Load Modules
module DroneLocalization

## Load Required Libraries
using DMUStudent
using ..Obfuscatee
using POMDPs
using StaticArrays
using POMDPTools
using Random
using Compose
using Nettle
using ProgressMeter
using JSON
import LinearAlgebra: SVector

export 
    DronePOMDP,
    DroneState,
    droneflight
    ## TODO - EXPORT ANYTHING ELSE THAT NEEDS TO BE EXPORTED
struct DroneState
    drone::SVector{3, Int} # x, y, z for the drone
    target::SVector{3, Int} # x, y, z for the target
    bystander::SVector{3, Int} # x, y, z for the bystander
    ## TODO - Think about making the ints into float64s
    ## TODO - Introduce any other state variables that are needed
end

## Conversions for the DroneState to make it compatible with the functions
Base.convert(::Type{Svector{9, Int}}, s::DroneState) = SA[s.drone..., s.target...]
Base.convert(::Type{AbstractVector{Int}}, s::DroneState) = convert(SVector{9, Int}, s)
Base.convert(::Type{AbstractVector}, s::DroneState) = convert(SVector{9, Int}, s)
Base.convert(::Type{AbstractArray}, s::DroneState) = convert(SVector{9, Int}, s)


## Define the Drone POMDP as a subtype of POMDP
struct DronePOMDP <: POMDP{DroneState, Symbol, SVector{6,Int}}
    size::SVector{3, Int} # x, y, z for the grid
    ## TODO MVP - For now z will always be one but think about introducing a 3D grid
    obstacles::Set{SVector{3, Int}} # x, y, z for the obstacles
    blocked::BitArray{2}
    drone_init::SVector{3, Int}
    obsindices::Array{Union{Nothing,Int}, 6} ##TODO - Figure out what this does/means (TB updated 04/13/2024 - I think this references the observation states)
end


## Define the observation states of the drone flight pomdp
function droneflight_observations(size)
    os = SVector{6,Int}[]
    for west in 0:size[1]-1
        for east in 0:size[1]-west-1
            for north in 0:size[2]-1
                for south in 0:size[2]-north-1
                    for up in 0:size[3]-1
                        for down in 0:size[3]-up-1
                            push!(os, SVector(east,west,north,south,up,down))
                        end
                    end
                end
            end
        end
    end
    return os
end

## Define the Drone POMDP
function DronePOMDP(;size=(10, 10, 1), n_obstacles=15, rng::AbstractRNG=Random.MersenneTwister(69)) ##TODO - Hash out these values
    obstacles = Set{SVector{3, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]), rand(rng, 1:size[3]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end
    drone_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]), rand(rng, 1:size[3]))

    obsindices = Array{Union{Nothing,Int}}(nothing, size[1], size[1], size[2], size[2], size[3], size[3]) ##TODO - FIgure out what this does/means (linked to the above TODO) (TB updated 04/13/2024 - I think is correct, but needs to be verified)
    for (ind, o) in enumerate(droneflight_observations(size))
        obsindices[(o.+1)...] = ind
    end

    DronePOMDP(size, obstacles, blocked, drone_init, obsindices)
end



end # module end