module twoDDroneLocalization


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
    twoDDronePOMDP,
    twoDDroneState,
    twoDdroneflight

struct twoDDroneState
    drone::SVector{2, Int}
    target::SVector{2, Int}
    bystander::SVector{2, Int}
end

Base.convert(::Type{SVector{6, Int}}, s::twoDDroneState) = SA[s.drone..., s.target..., s.bystander...]
Base.convert(::Type{AbstractVector{Int}}, s::twoDDroneState) = convert(SVector{6, Int}, s)
Base.convert(::Type{AbstractVector}, s::twoDDroneState) = convert(SVector{6, Int}, s)
Base.convert(::Type{AbstractArray}, s::twoDDroneState) = convert(SVector{6, Int}, s)


struct twoDDronePOMDP <: POMDP{twoDDroneState, Symbol, SVector{4,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    drone_init::SVector{2, Int}
    obsindices::Array{Union{Nothing,Int}, 4}
end

function twoDdroneflight_observations(size)
    os = SVector{4,Int}[]
    for left in 0:size[1]-1
        for right in 0:size[1]-left-1
            for up in 0:size[2]-1
                for down in 0:size[2]-up-1
                    push!(os, SVector(left, right, up, down))
                end
            end
        end
    end
    return os
end

function twoDDronePOMDP(;size=(10, 10), n_obstacles=12, rng::AbstractRNG=Random.MersenneTwister(20))
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end
    drone_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))

    obsindices = Array{Union{Nothing,Int}}(nothing, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(twoDdroneflight_observations(size))
        obsindices[(o.+1)...] = ind
    end

    twoDDronePOMDP(size, obstacles, blocked, drone_init, obsindices)
end


Random.rand(rng::AbstractRNG, ::Random.SamplerType{twoDDronePOMDP}) = twoDDronePOMDP(rng=rng)

POMDPs.actions(m::twoDDronePOMDP) = (:left, :right, :up, :down, :measure)
POMDPs.states(m::twoDDronePOMDP) = vec(collect(twoDDroneState(SVector(c[1],c[2]), SVector(c[3], c[4]), SVector(c[5], c[6])) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])))
POMDPs.observations(m::twoDDronePOMDP) = twoDdroneflight_observations(m.size)
POMDPs.discount(m::twoDDronePOMDP) = 0.95

POMDPs.stateindex(m::twoDDronePOMDP, s) = LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2]))[s.drone..., s.target..., s.bystander...]

POMDPs.actionindex(m::twoDDronePOMDP, a) = actionind[a]
POMDPs.obsindex(m::twoDDronePOMDP, o) = m.obsindices[(o.+1)...]::Int

const actiondir = Dict(:left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1), :measure=>SVector(0,0))
const actionind = Dict(:left=>1, :right=>2, :up=>3, :down=>4, :measure=>5)

function bounce(m::twoDDronePOMDP, pos, change)
    new = clamp.(pos + change, SVector(1,1), m.size)
    if m.blocked[new[1], new[2]]
        return pos
    else
        return new
    end
end

# drone moves deterministically
# target usually moves randomly, but moves away if near
# bystander moves randomly
function POMDPs.transition(m::twoDDronePOMDP, s, a)
    newdrone = bounce(m, s.drone, actiondir[a])

    if isterminal(m, s)
        @assert s.drone == s.target
        # return a new terminal state where the drone has moved
        # this maintains the property that the drone always moves the same, regardless of the target and bystander states
        return SparseCat([twoDDroneState(newdrone, newdrone, s.bystander)], [1.0])
    end


    targets = [s.target]
    targetprobs = Float64[0.0]

    if sum(abs, newdrone - s.target) > 2 # move randomly
        for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
            newtarget = bounce(m, s.target, change)
            if newtarget == s.target
                targetprobs[1] += 0.25
            else
                push!(targets, newtarget)
                push!(targetprobs, 0.25)
            end
        end
    else # move away 
        away = sign.(s.target - s.drone)
        if sum(abs, away) == 2 # diagonal
            away = away - SVector(0, away[2]) # preference to move in x direction
        end
        newtarget = bounce(m, s.target, away)
        targets[1] = newtarget
        targetprobs[1] = 1.0
    end

    bystanders = [s.bystander]
    bystanderprobs = Float64[0.0]
    for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
        newbystander = bounce(m, s.bystander, change)
        if newbystander == s.bystander
            bystanderprobs[1] += 0.25
        else
            push!(bystanders, newbystander)
            push!(bystanderprobs, 0.25)
        end
    end

    states = twoDDroneState[]    
    probs = Float64[]
    for (t, tp) in zip(targets, targetprobs)
        for (w, wp) in zip(bystanders, bystanderprobs)
            push!(states, twoDDroneState(newdrone, t, w))
            push!(probs, tp*wp)
        end
    end

    return SparseCat(states, probs)
end

POMDPs.isterminal(m::twoDDronePOMDP, s) = s.target == s.drone

function POMDPs.observation(m::twoDDronePOMDP, a, sp)
    left = sp.drone[1]-1
    right = m.size[1]-sp.drone[1]
    up = m.size[2]-sp.drone[2]
    down = sp.drone[2]-1
    ranges = SVector(left, right, up, down)
    for obstacle in m.obstacles
        ranges = laserbounce(ranges, sp.drone, obstacle)
    end
    ranges = laserbounce(ranges, sp.drone, sp.target)
    ranges = laserbounce(ranges, sp.drone, sp.bystander)
    os = SVector(ranges, SVector(0, 0, 0, 0))
    if all(ranges.==0.0) || a == :measure
        probs = SVector(1.0, 0.0)
    else
        probs = SVector(0.1, 0.9)
    end
    return SparseCat(os, probs)
end

function laserbounce(ranges, drone, obstacle)
    left, right, up, down = ranges
    diff = obstacle - drone
    if diff[1] == 0
        if diff[2] > 0
            up = min(up, diff[2]-1)
        elseif diff[2] < 0
            down = min(down, -diff[2]-1)
        end
    elseif diff[2] == 0
        if diff[1] > 0
            right = min(right, diff[1]-1)
        elseif diff[1] < 0
            left = min(left, -diff[1]-1)
        end
    end
    return SVector(left, right, up, down)
end

function POMDPs.initialstate(m::twoDDronePOMDP)
    return Uniform(twoDDroneState(m.drone_init, SVector(x, y), SVector(x,y)) for x in 1:m.size[1], y in 1:m.size[2])
end

function POMDPTools.render(m::twoDDronePOMDP, step)
    nx, ny = m.size
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
        cell = cell_ctx((x,y), m.size)
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
        drone_ctx = cell_ctx(step[:sp].drone, m.size)
        drone = compose(drone_ctx, circle(0.5, 0.5, 0.5), fill("green"))
        target_ctx = cell_ctx(step[:sp].target, m.size)
        target = compose(target_ctx, circle(0.5, 0.5, 0.5), fill("orange"))
        bystander_ctx = cell_ctx(step[:sp].bystander, m.size)
        bystander = compose(bystander_ctx, circle(0.5, 0.5, 0.5), fill("purple"))
    else
        drone = nothing
        target = nothing
        bystander = nothing
    end

    if haskey(step, :o) && haskey(step, :sp)
        o = step[:o]
        drone_ctx = cell_ctx(step[:sp].drone, m.size)
        left = compose(context(), line([(0.0, 0.5),(-o[1],0.5)]))
        right = compose(context(), line([(1.0, 0.5),(1.0+o[2],0.5)]))
        up = compose(context(), line([(0.5, 0.0),(0.5, -o[3])]))
        down = compose(context(), line([(0.5, 1.0),(0.5, 1.0+o[4])]))
        lasers = compose(drone_ctx, strokedash([1mm]), stroke("red"), left, right, up, down)
    else
        lasers = nothing
    end

    sz = min(w,h)
    return compose(context((w-sz)/2, (h-sz)/2, sz, sz), drone, target, bystander, lasers, grid, outline)
end

function POMDPs.reward(m::twoDDronePOMDP, s, a, sp)
    if isterminal(m, s)
        return 0.0
    elseif sp.drone == sp.target
        return 200.0
    elseif sp.drone == sp.bystander
        return -10.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end

function POMDPs.reward(m::twoDDronePOMDP, s, a)
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
    return context((x-1)/nx, (ny-y)/ny, 1/nx, 1/ny)
end

@binclude(".bin/hw6_eval")

end
