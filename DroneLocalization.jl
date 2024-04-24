##Easy to call python from julia, good for visualization.

## Reference: HW6.jl 
## Load Modules
module DroneLocalization

## Load Required Libraries
using DMUStudent
using POMDPs
using StaticArrays
using POMDPTools
using Random
using Compose
using Nettle
using ProgressMeter
using JSON



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
Base.convert(::Type{SVector{9, Int}}, s::DroneState) = SA[s.drone..., s.target...]
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


Random.rand(rng::AbstractRNG, p::Random.SamplerType{DronePOMDP}) = DronePOMDP(rng=rng)

POMDPs.actions(m::DronePOMDP) = (:forward, :backward, :left, :right, :up, :down, :measure)
POMDPs.states(m::DronePOMDP) = vec(collect(DroneState(SVector(c[1],c[2],c[3]), SVector(c[4], c[5],c[6]), SVector(c[7], c[8], c[9])) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[3], 1:m.size[1], 1:m.size[2], 1:m.size[3], 1:m.size[1], 1:m.size[2], 1:m.size[3])))
POMDPs.observations(m::DronePOMDP) = droneflight_observations(m.size)
POMDPs.discount(m::DronePOMDP) = 0.95 ##TODO - Define What we want this to be

POMDPs.stateindex(m::DronePOMDP, s) = LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[3], 1:m.size[1], 1:m.size[2], 1:m.size[3], 1:m.size[1], 1:m.size[2], 1:m.size[3]))[s.drone..., s.target..., s.bystander...]

POMDPs.actionindex(m::DronePOMDP, a) = actionind[a]
POMDPs.obsindex(m::DronePOMDP, o) = m.obsindices[(o.+1)...]::Int


## Define the actions for the drone
const actiondir = Dict(:forward=>SVector(1,0,0), :backward=>SVector(-1,0,0), :left=>SVector(0,-1,0), :right=>SVector(0,1,0), :up=>SVector(0,0,1), :down=>SVector(0,0,-1), :measure=>SVector(0,0,0))
const actionind = Dict(:forward=>1, :backward=>2, :left=>3, :right=>4, :up=>5, :down=>6, :measure=>7)

## Define the behavior function for the drone when it is blocked and not blocked
function bounce(m::DronePOMDP, pos, change)
    new = clamp.(pos + change, SVector(1,1,1), m.size)
    if m.blocked[new[1], new[2], new[3]]
        return pos
    else
        return new
    end
end 


# drone moves deterministically
# target usually moves randomly, but moves away if near
# bystander moves randomly

# Define the transition functions
function POMDPs.transition(m::DronePOMDP, s, a)
    newdrone = bounce(m, s.drone, actiondir[a])

    if isterminal(m,s)
        @assert s.drone == s.target
        # return a new terminal state where the drone has moved
        # this maintains the property that the drone always moves the same, regardless of the target and bystander states
        return SparseCat([LTState(newdrone, newdrone, s.bystander)], [1.0])
    end

    ## Target transtition Probabilities
    targets = [s.target]
    targetprobs = Float64[0.0] #0.0, because the target cannot be reachable initially
    ## Define the random movement of the target TODO - Solidify This
    ##TODO - I feel like we might want this to move randomly in the x and y direction always.
    # move randomly
    
    for change in (SVector(1,0,0), SVector(-1,0,0), SVector(0,-1,0), SVector(0,1,0), SVector(0,0,1), SVector(0,0,-1))
        newtarget = bounce(m, s.target, change)
        if newtarget == s.target
            targetprobs[1] += Float64(1/6)
        else
            push!(targets, newtarget)
            push!(targetprobs, Float64(1/6))
        end
    end
    
    ##This aspect is probably not necessary
    # else # move away 
    #     away = sign.(s.target - s.drone)
    #     if sum(abs, away) == 2 # diagonal
    #         away = away - SVector(0, away[2]) # preference to move in x direction
    #     end
    #     newtarget = bounce(m, s.target, away)
    #     targets[1] = newtarget
    #     targetprobs[1] = 1.0
    # end

    ## Bystander Transition Probabilities
    bystanders = [s.bystander]
    bystanderprobs = Float64[0.0]
    for change in (SVector(1,0,0), SVector(-1,0,0), SVector(0,-1,0), SVector(0,1,0), SVector(0,0,1), SVector(0,0,-1))
        newbystander = bounce(m, s.bystander, change)
        if newbystander == s.bystander
            bystanderprobs[1] += Float64(1/6)
        else
            push!(bystanders, newbystander)
            push!(bystanderprobs, Float64(1/6))
        end
    end

    
    states = DroneState[]    
    probs = Float64[]
    for (t, tp) in zip(targets, targetprobs)
        for (w, wp) in zip(bystanders, bystanderprobs)
            push!(states, DroneState(newdrone, t, w))
            push!(probs, tp*wp)
        end
    end

    return SparseCat(states, probs)
end
    
POMDPs.isterminal(m::DronePOMDP, s) = s.target == s.drone

## Define the observation function
## The observation capabilities of the drone are:
    ## Forward: Lidar can detect the distance to the nearest obstacle in the forward direction: Confidence - 0.7
    ## Backward: Lidar can detect the distance to the nearest obstacle in the backward direction: Confidence - 0.7
    ## Left: Lidar can detect the distance to the nearest obstacle in the left direction: Confidence - 0.7
    ## Right: Lidar can detect the distance to the nearest obstacle in the right direction: Confidence - 0.7
    ## Down: Camera can detect the distance to the nearest obstacle in the down direction: Confidence - 0.9
    ## Up: No Sensors

function POMDPs.observation(m::DronePOMDP, a, sp)
    ## Calculating the distance from (0,0,0) to the drone 
    ## +x:forward, -x:backward, +y:right, -y:left, +z:down, -z:up
    forward = m.size[1]-sp.drone[1]
    backward = sp.drone[1]-1
    left = sp.drone[2]-1
    right = m.size[2]-sp.drone[2]
    up = sp.drone[3]-1
    down = m.size[3]-sp.drone[3]
    ranges = SVector(forward, backward, left, right, up, down)
    for obstacle in m.obstacles
        ranges = sensorbounce(ranges,sp.drone,obstacle)
    end
    ranges = sensorbounce(ranges,sp.drone,sp.target)
    ranges = sensorbounce(ranges,sp.drone,sp.bystander)

    os = SVector(ranges, SVector(0, 0, 0, 0, 0, 0))
    if all(ranges.==0.0) || a == :measure
        probs = SVector(1.0, 0.0)
    else
        probs = SVector(0.1, 0.9) ## TODO Have Sunberg Explain Probabilities(ranges, NULL)
    end

    return SparseCat(os, probs)
end

function sensorbounce(ranges,drone,obstacle)
    forward, backward, left, right, up, down = ranges
    diff = obstacle - drone
    if diff[1] == 0
        if diff[2] > 0
            right = min(right, diff[2]-1)
        elseif diff[2] < 0
            left = min(left, -diff[2]-1)
        end
        if diff[3] > 0
            down = min(down, diff[3]-1)
        elseif diff[3] < 0
            up = min(up, -diff[3]-1)
        end
    elseif diff[2] == 0
        if diff[1] > 0
            forward = min(forward, diff[1]-1)
        elseif diff[1] < 0
            backward = min(backward, -diff[1]-1)
        end
        if diff[3] > 0
            down = min(down, diff[3]-1)
        elseif diff[3] < 0
            up = min(up, -diff[3]-1)
        end
    elseif diff[3] == 0
        if diff[1] > 0
            forward = min(forward, diff[1]-1)
        elseif diff[1] < 0
            backward = min(backward, -diff[1]-1)
        end
        if diff[2] > 0
            right = min(right, diff[2]-1)
        elseif diff[2] < 0
            left = min(left, -diff[2]-1)
        end    
    end
    return SVector(forward, backward, left, right, up, down)
end

function POMDPs.initialstate(m::DronePOMDP)
    return Uniform(DroneState(m.drone_init, SVector(x,y,z), SVector(x,y,z)) for x in 1:m.size[1], y in 1:m.size[2], z in 1:m.size[3])
end

## TODO - see if possible better render function 3D
## TODO - Have SUNBERG explain this
## TODO - This could be improved
## Easiest to do a 3D plot likely, interactive and rotatable - LOOK INTO THIS
            # PLOTS.JL
            # MULTIBODY.JL

function POMDPTools.render(m::DronePOMDP, step)

    ############### XY PLANE ################
    nx, ny = m.size[1:2]
    cells = []
    target_marginal = zeros(nx, ny)
    bystander_marginal = zeros(nx, ny)
    if haskey(step, :bp) && !ismissing(step[:bp])
        for sp in support(step[:bp])
            p = pdf(step[:bp], sp)
            target_marginal[sp.target...] += p
            bystander_marginal[sp.bystander...] += p
        end
    end

    for x in 1:nx, y in 1:ny
        cell = cell_ctx((x,y), m.size[1:2])
        if SVector(x, y) in m.obstacles
            compose!(cell, rectangle(), fill("darkgray"))
        else
            w_op = sqrt(bystander_marginal[x, y])
            w_rect = compose(context(), rectangle(), fillopacity(w_op), fill("lightblue"), stroke("gray"))
            t_op = sqrt(target_marginal[x, y])
            t_rect = compose(context(), rectangle(), fillopacity(t_op), fill("yellow"), stroke("gray"))
            compose!(cell, w_rect, t_rect)
        end
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), cells...)
    outline = compose(context(), linewidth(1mm), rectangle(), fill("white"), stroke("gray"))

    if haskey(step, :sp)
        drone_ctx = cell_ctx(step[:sp].drone, m.size[1:2])
        drone = compose(drone_ctx, circle(0.5, 0.5, 0.5), fill("green"))
        target_ctx = cell_ctx(step[:sp].target, m.size[1:2])
        target = compose(target_ctx, circle(0.5, 0.5, 0.5), fill("orange"))
        bystander_ctx = cell_ctx(step[:sp].bystander, m.size[1:2])
        bystander = compose(bystander_ctx, circle(0.5, 0.5, 0.5), fill("purple"))
    else
        drone = nothing
        target = nothing
        bystander = nothing
    end

    if haskey(step, :o) && haskey(step, :sp)
        o = step[:o]
        drone_ctx = cell_ctx(step[:sp].drone, m.size[1:2])
        left = compose(context(), line([(0.0, 0.5),(-o[1],0.5)]))
        right = compose(context(), line([(1.0, 0.5),(1.0+o[2],0.5)]))
        up = compose(context(), line([(0.5, 0.0),(0.5, -o[3])]))
        down = compose(context(), line([(0.5, 1.0),(0.5, 1.0+o[4])]))
        lidar = compose(drone_ctx, strokedash([1mm]), stroke("red"), left, right, up, down)
    else
        lidar = nothing
    end

    sz = min(w,h)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), drone, target, bystander, lidar, grid, outline)
end

function POMDPs.reward(m::DronePOMDP, s, a, sp)
    if isterminal(m, sp)
        return 0.0
    elseif sp.drone == sp.target
        return 200.0
    elseif sp.drone == sp.bystander
        return -50.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end

function POMDPs.reward(m::DronePOMDP, s, a)
    r = 0.0
    td = transition(m, s, a)
    for (sp, w) in weighted_iterator(td)
        r += w*reward(m, s, a, sp)
    end
    return r    
end

function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return compose(context((x-1)/nx, (y-1)/ny, 1/nx, 1/ny))
end

end # module end