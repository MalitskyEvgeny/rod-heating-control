Base.@kwdef struct Problem{Ff, Fφ, Fy}
    L
    T
    a²
    ν
    f::Ff
    φ::Fφ
    y::Fy
end

struct Grid
    h::Float64
    τ::Float64
    Nₓ::Int
    Nₜ::Int
    s::Vector{Float64} 
    t::Vector{Float64}
end

function Grid(prob::Problem, Nₓ::Int, Nₜ::Int)
    h = prob.L / (Nₓ - 1)
    τ = prob.T / (Nₜ - 1)
    s  = collect(range(0, prob.L, length = Nₓ))
    t  = collect(range(0, prob.T, length = Nₜ))
    return Grid(h, τ, Nₓ, Nₜ, s, t)
end

struct Control
    p::Vector{Float64}
end