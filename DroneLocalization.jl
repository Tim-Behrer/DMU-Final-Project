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
    DroneState
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

struct DronePOMDP <: POMDP{DroneState, Symbol, SVector{4,Int}}
    size::SVector{3, Int} # x, y, z for the grid
    ## TODO MVP - For now z will always be one but think about introducing a 3D grid
    obstacles::Set{SVector{3, Int}} # x, y, z for the obstacles
    blocked::BitArray{2}
    drone_init::SVector{3, Int}
    obsindices::Array{Union{Nothing,Int}, 4} ##TODO - FIgure out what this does/means
end






## Create and Define the Drone POMDP Environment
# TODO - Define the Drone POMDP Environment minimum viable product (QuicckPOMDP Definition)

end # module end