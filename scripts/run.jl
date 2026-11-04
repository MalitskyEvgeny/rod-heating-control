using LinearAlgebra

using HeatControl

using LinearAlgebra, Random, Plots

# --- Общий явный шаг: out = B*u (используется и для x, и для ψ) ---
function explicit_step!(out, u, μ, νh)
    N = length(u)
    @inbounds for i in 2:N-1
        out[i] = u[i] + μ * (u[i-1] - 2u[i] + u[i+1])
    end
    out[1] = (1 - 2μ) * u[1] + 2μ * u[2]
    out[N] = 2μ * u[N-1] + (1 - 2μ * (1 + νh)) * u[N]
    return out
end

function build_operator(P, grid)
    h, τ, ν, N = grid.h, grid.τ, P.ν, grid.Nₓ
    μ = P.a² * τ / (2h^2)
    A = Tridiagonal(fill(-μ, N-1), fill(1 + 2μ, N), fill(-μ, N-1))
    A.du[1]  *= 2
    A.dl[end] *= 2
    A.d[N]   += 2ν * h * μ
    return μ, lu(A)
end

# --- Прямая задача: возвращает x(·,T) ---
function solve_forward(P, grid, p, μ, F)
    h, τ, ν, N = grid.h, grid.τ, P.ν, grid.Nₓ
    x   = P.φ.(grid.s)
    rhs = similar(x)
    for i in 1:grid.Nₜ-1
        explicit_step!(rhs, x, μ, ν*h)
        tσ = grid.t[i] + τ/2
        @inbounds for j in 1:N
            rhs[j] += τ * P.f(grid.s[j], tσ)
        end
        rhs[N] += 2ν * h * μ * (p[i] + p[i+1])
        copyto!(x, rhs)
        ldiv!(F, x)
    end
    return x
end

# --- Сопряжённая задача: возвращает ψ(L,t) ---
function solve_adjoint(P, grid, xT, μ, F)
    h, ν = grid.h, P.ν
    ψ   = 2 .* (xT .- P.y.(grid.s))
    rhs = similar(ψ)
    ψ_L = zeros(grid.Nₜ)
    ψ_L[end] = ψ[end]
    for i in grid.Nₜ-1:-1:1
        explicit_step!(rhs, ψ, μ, ν*h)
        copyto!(ψ, rhs)
        ldiv!(F, ψ)
        ψ_L[i] = ψ[end]
    end
    return ψ_L
end

# --- Функционал (трапеции по s) ---
function cost(P, grid, xT)
    r = xT .- P.y.(grid.s)
    return grid.h * (sum(abs2, r) - 0.5*(r[1]^2 + r[end]^2))
end

# --- Функционал + градиент ---
function cost_and_grad(P, grid, p, μ, F)
    xT  = solve_forward(P, grid, p, μ, F)
    ψ_L = solve_adjoint(P, grid, xT, μ, F)
    return cost(P, grid, xT), P.a² * P.ν .* ψ_L, xT
end

weights(grid) = (w = fill(grid.τ, grid.Nₜ); w[1] = w[end] = grid.τ/2; w)

ip(u, v, w)  = sum(w .* u .* v)         # скалярное произведение в L²(0,T)
nrm2(u, w)   = ip(u, u, w)

# решение линейной подзадачи на множестве pmin ≤ p ≤ pmax
lmo(g, pmin, pmax) = ifelse.(g .> 0, pmin, pmax)

# оценка константы Липшица градиента (степенной метод по разности градиентов)
function estimate_lipschitz(P, grid, μ, F, w; iters = 30, seed = 1)
    rng = MersenneTwister(seed)
    p0  = zeros(grid.Nₜ)
    _, g0 = cost_and_grad(P, grid, p0, μ, F)
    d = randn(rng, grid.Nₜ);  d ./= sqrt(nrm2(d, w))
    λ = 0.0
    for _ in 1:iters
        _, g1 = cost_and_grad(P, grid, p0 .+ d, μ, F)
        Hd = g1 .- g0
        λ  = sqrt(nrm2(Hd, w))              # ‖d‖_w = 1
        λ == 0 && break
        d  = Hd ./ λ
    end
    return λ
end

function plot_state(P, grid, p, xT, hist, k; title_prefix = "")
    p1 = plot(grid.s, xT; lw = 2, label = "x(s,T)", xlabel = "s")
    plot!(p1, grid.s, P.y.(grid.s); lw = 2, ls = :dash, label = "y(s)")
    title!(p1, "$title_prefix iter $k")

    p2 = plot(grid.t, p; lw = 2, label = "p(t)", xlabel = "t")

    p3 = plot(hist.nfwd, hist.J; yscale = :log10, lw = 2, label = "J",
              xlabel = "число прямых задач")

    plt = plot(p1, p2, p3; layout = (1, 3), size = (1300, 350))
    display(plt)
    return plt
end
function frank_wolfe(P, grid; rule::Symbol = :halving,
                     pmin = -1e6, pmax = 1e6, p0 = zeros(grid.Nₜ),
                     iters = 200, tol = 1e-8,
                     ρ = nothing, ε = 1e-3, ε0 = 1e-3, Lip = nothing,
                     plot_every = 10, save_plots = false,
                     make_gif = true, outdir = "results")
    
    mkpath(outdir)
    anim = Animation()

    μ, F = build_operator(P, grid)
    w = weights(grid)

    if rule == :lipschitz
        Lip === nothing && (Lip = estimate_lipschitz(P, grid, μ, F, w))
        ρ === nothing && (ρ = 1 / (Lip + 2ε))
        @assert ε0 ≤ ρ ≤ 2 / (Lip + 2ε) "ρ вне допустимого диапазона"
        @info "L = $Lip, ρ = $ρ"
    end

    p = copy(p0)
    J, g, xT = cost_and_grad(P, grid, p, μ, F)
    hist = (J = Float64[], gap = Float64[], α = Float64[], nfwd = Int[])
    nfwd = 1
    push!(hist.J, J); push!(hist.nfwd, nfwd)

    if plot_every > 0
        plt = plot_state(P, grid, p, xT, hist, 0; title_prefix = string(rule))
        make_gif && frame(anim, plt)
    end

    for k in 1:iters
        p̄     = lmo(g, pmin, pmax)
        dir   = p̄ .- p
        slope = ip(g, dir, w)               # ⟨J', p̄ − p⟩ ≤ 0
        gap   = -slope
        gap ≤ tol && break

        if rule == :halving                  # способ 2
            α = 1.0
            while true
                p_new = p .+ α .* dir
                J_new = cost(P, grid, solve_forward(P, grid, p_new, μ, F))
                nfwd += 1
                J_new < J && break
                α /= 2
                α < 1e-14 && (α = 0.0; break)
            end
            α == 0 && break
        else                                 # способ 3
            α = min(1.0, ρ * abs(slope) / nrm2(dir, w))
        end

        p .+= α .* dir
        J, g, xT = cost_and_grad(P, grid, p, μ, F)
        nfwd += 1
        push!(hist.J, J); push!(hist.gap, gap); push!(hist.α, α); push!(hist.nfwd, nfwd)
        @info "iter $k" J gap α

        if plot_every > 0 && k % plot_every == 0
            plt = plot_state(P, grid, p, xT, hist, k; title_prefix = string(rule))
            make_gif   && frame(anim, plt)
            save_plots && savefig(plt, joinpath(outdir, "fw_$(rule)_iter$(lpad(k, 4, '0')).png"))
        end
    end

    if plot_every > 0
        plt = plot_state(P, grid, p, xT, hist, length(hist.J) - 1;
                         title_prefix = string(rule) * " (final)")
        make_gif   && frame(anim, plt)
        savefig(plt, joinpath(outdir, "fw_$(rule)_final.png"))
        make_gif   && gif(anim, joinpath(outdir, "fw_$(rule).gif"); fps = 2)
    end
    return p, hist
end

grid = Grid(P1, 4, 800)
filname = "P_24x800_750_10&10/P9"

p2, h2 = frank_wolfe(P1, grid; rule = :halving,   pmin = -10, pmax = 10,
                     iters = 750, tol = 1e-6, plot_every = 10, outdir = filname, Lip=P1.a²*P1.ν/2)
p3, h3 = frank_wolfe(P1, grid; rule = :lipschitz, pmin = -10, pmax = 10,
                     iters = 750, tol = 1e-6, plot_every = 10, outdir = filname, Lip=P1.a²*P1.ν/2)


cmp = plot(h2.nfwd, h2.J; yscale = :log10, lw = 2, label = "правило 2 (дробление)",
           xlabel = "число прямых задач", ylabel = "J")
plot!(cmp, h3.nfwd, h3.J; lw = 2, label = "правило 3 (Липшиц)")
savefig(cmp, filname * "/compare_J.png")
