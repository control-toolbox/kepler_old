# 2bp.jl

using OptimalControl
using NLPModelsIpopt
using OrdinaryDiffEq
using Plots
using MINPACK
using ForwardDiff
using LinearAlgebra

## Problem definition. Mass in kg, distance in Mm, time in hours (h).

Tmax = 60                                  # Maximum thrust (Newtons)
cTmax = 3600^2 / 1e6; T = Tmax * cTmax     # Conversion from Newtons to kg x Mm / h²
mass0 = 1500                               # Initial mass of the spacecraft
β = 1.42e-02                               # Engine specific impulsion
μ = 5165.8620912                           # Earth gravitation constant
x0 = [11.625, 0.75, 0, 6.12e-02, 0, π]     # Initial state (fixed initial longitude)
yf = [42.165, 0, 0, 0, 0]                  # Final state (free final longitude)

function F0(x::Vector{T}) where T
    P, ex, ey, hx, hy, L = x
    pdm = sqrt(P / μ)
    cl = cos(L)
    sl = sin(L)
    w = 1 + ex * cl + ey * sl
    F = zeros(T, 6)
    F[6] = w^2 / (P * pdm)
    return F
end

function F1(x::Vector{T}) where T
    P, ex, ey, hx, hy, L = x
    pdm = sqrt(P/μ)
    cl = cos(L)
    sl = sin(L)
    F = zeros(T, 6)
    F[2] = pdm *   sl
    F[3] = pdm * (-cl)
    return F
end

function F2(x::Vector{T}) where T
    P, ex, ey, hx, hy, L = x
    pdm = sqrt(P/μ)
    cl = cos(L)
    sl = sin(L)
    w = 1 + ex * cl + ey * sl
    F = zeros(T, 6)
    F[1] = pdm * 2 * P / w
    F[2] = pdm * (cl + (ex + cl) / w)
    F[3] = pdm * (sl + (ey + sl) / w)
    return F
end

function F3(x::Vector{T}) where T
    P, ex, ey, hx, hy, L = x
    pdm = sqrt(P/μ)
    cl = cos(L)
    sl = sin(L)
    w = 1 + ex * cl + ey * sl
    pdmw = pdm / w
    zz = hx * sl - hy * cl
    uh = (1 + hx^2 + hy^2) / 2
    F = zeros(T, 6)
    F[2] = pdmw * (-zz * ey)
    F[3] = pdmw *   zz * ex
    F[4] = pdmw *   uh * cl
    F[5] = pdmw *   uh * sl
    F[6] = pdmw *   zz
    return F
end

@def ocp begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x = (P, ex, ey, hx, hy, L) ∈ R⁶, state
    u ∈ R³, control
    x(0) == x0
    x[1:5](tf) == yf
    mass = mass0 - β * T * t
    ẋ(t) == F0(x(t)) + T / mass * (u₁(t) * F1(x(t)) + u₂(t) * F2(x(t)) + u₃(t) * F3(x(t)))
    -[1, 1, 1] ≤ u(t) ≤ [1, 1, 1]
    u₁(t)^2 + u₂(t)^2 + u₃(t)^2 ≤ 1
    .8 * x0[1] ≤ P(t) ≤ 1.2 * x0[1]
    -1 ≤ ex(t) ≤ 1
    -1 ≤ ey(t) ≤ 1
    -1 ≤ hx(t) ≤ 1
    -1 ≤ hy(t) ≤ 1
    0 ≤ L(t) ≤ 2.5π
    tf → min
end

## Initialisations

init = Dict{Real, Tuple{Real, Vector{Real}}}()
#tf = 15.2055; p0 =-[.361266, 22.2412, 7.87736, 0, 0, -5.90802]; init[60] = (tf, p0)
tf = 14.79640450270989; p0 = [-0.361266, -22.2412, -7.87736, -0.0, -0.0, 5.90802]; init[60] = (tf, p0)
tf = 1.320e2; p0 =-[-4.743728539366440e+00, -7.171314869854240e+01, -2.750468309804530e+00, 4.505679923365745e+01, -3.026794475592510e+00, 2.248091067047670e+00]; init[6] = (tf, p0)
tf = 1.210e3; p0 =-[-2.215319700438820e+01, -4.347109477345140e+01, 9.613188807286992e-01, 3.181800985503019e+02, -2.307236094862410e+00, -5.797863110671591e-01]; init[0.7] = (tf, p0)
tf = 6.080e3; p0 =-[-1.234155379067110e+02, -6.207170881591489e+02, 5.742554220129187e-01, 1.629324243017332e+03, -2.373935935351530e+00, -2.854066853269850e-01]; init[0.14] = (tf, p0)

## Direct

tf = init[Tmax][1] # debug: update init
#nlp_init = OCPInit(variable=tf)
#nlp_sol = solve(ocp; init=nlp_init, grid_size=100) # to be tested

## Shooting (1/2)

function ur(t, x, p, tf) # Regular maximising control 
    H1 = p' * F1(x)
    H2 = p' * F2(x)
    H3 = p' * F3(x)
    u = [H1, H2, H3]
    u = u / sqrt(u[1]^2 + u[2]^2 + u[3]^2)
    return u
end

fr = Flow(ocp, ur) # Regular flow (first version)

function shoot(ξ::Vector{T}) where T
    tf, p0 = ξ[1], ξ[2:end]
    xf, pf = fr(0, x0, p0, tf)
    s = zeros(T, 7)
    s[1:5] = xf[1:5] - yf
    s[6] = pf[6]
    s[7] = p0[1]^2 + p0[2]^2 + p0[3]^2 + p0[4]^2 + p0[5]^2 + p0[6]^2 - 1
    return s
end

## Solve

tf, p0 = init[Tmax]
p0 = p0 / norm(p0) # Normalization |p0|=1 for free final time
ξ = [tf; p0]; # Initial guess
jshoot(ξ) = ForwardDiff.jacobian(shoot, ξ)
shoot!(s, ξ) = (s[:] = shoot(ξ); nothing)
jshoot!(js, ξ) = (js[:] = jshoot(ξ); nothing)
@time bvp_sol = fsolve(shoot!, jshoot!, ξ, show_trace=true); println(bvp_sol)

## Shooting (2/2)

hr = (t, x, p) -> begin # Regular maximised Hamiltonian (more efficient)
    H0 = p' * F0(x)
    H1 = p' * F1(x)
    H2 = p' * F2(x)
    H3 = p' * F3(x)
    mass = mass0 - β*T*t
    h = H0 + T / mass * sqrt(H1^2 + H2^2 + H3^2) 
    return h
end

hr = Hamiltonian(hr; autonomous=false)
fr = Flow(hr) # Regular flow (again)
@time bvp_sol = fsolve(shoot!, jshoot!, ξ, show_trace=true); println(bvp_sol)
tf = bvp_sol.x[1]; p0 = bvp_sol.x[2:end]

## Plots

ode_sol = fr((0, tf), x0, p0)
t  = ode_sol.t; N = size(t, 1)
P  = ode_sol[1, :]
ex = ode_sol[2, :]
ey = ode_sol[3, :]
hx = ode_sol[4, :]
hy = ode_sol[5, :]
L  = ode_sol[6, :]
cL = cos.(L)
sL = sin.(L)
w  = @. 1 + ex * cL + ey * sL
Z  = @. hx * sL - hy * cL
C  = @. 1 + hx^2 + hy^2
q1 = @. P *((1 + hx^2 - hy^2) * cL + 2 * hx * hy * sL) / (C * w)
q2 = @. P *((1 - hx^2 + hy^2) * sL + 2 * hx * hy * cL) / (C * w)
q3 = @. 2 * P * Z / (C * w)

plt1 = plot3d(1; xlim = (-60, 60), ylim = (-60, 60), zlim = (-5, 5), title = "Orbit transfer", legend=false)
@gif for i = 1:N
    push!(plt1, q1[i], q2[i], q3[i])
end every N ÷ min(N, 100) 