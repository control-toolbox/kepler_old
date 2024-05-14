# Kepler.jl
# todo: vectorise plot exprs with @.

using OrdinaryDiffEq, ForwardDiff, MINPACK, LinearAlgebra, Plots

## Auxiliary flow definition

grad(f, x) = ForwardDiff.gradient(f, x)
jac(f, x) = ForwardDiff.jacobian(f, x)

function Flow(h; abstol=1e-12, reltol=1e-12)

    function rhs!(dz, z, dummy, t)
        n = size(z, 1) ÷ 2
        foo = z -> h(t, z[1:n], z[n+1:2*n])
        dh = grad(foo, z)
        dz[1:n] = dh[n+1:2n]
        dz[n+1:2n] = -dh[1:n]
    end
    
    function f(tspan, x0, p0; abstol=abstol, reltol=reltol, saveat=[])
        z0 = [ x0; p0 ]
        ode = ODEProblem(rhs!, z0, tspan)
        sol = OrdinaryDiffEq.solve(ode, Tsit5(), abstol=abstol, reltol=reltol, saveat=saveat)
        return sol
    end
    
    function f(t0, x0, p0, tf; abstol=abstol, reltol=reltol, saveat=[])
        sol = f((t0, tf), x0, p0, abstol=abstol, reltol=reltol, saveat=saveat)
        n = size(x0, 1)
        return sol[1:n, end], sol[n+1:2*n, end]
    end
    
    return f

end

## Problem definition

cTmax = (3600^2) / 1e6                                              # Conversion from Newtons
mass0 = 1500                                                        # Initial mass of the spacecraft
β = 1.42e-02                                                        # Engine specific impulsion
μ = 5165.8620912                                                    # Earth gravitation constant
t0 = 0                                                              # Initial time (final time is free)
x0 = [ 11.625, 0.75, 0, 6.12e-02, 0, π ]         # Initial state (fixed initial longitude)
xf_fixed = [ 42.165, 0, 0, 0, 0 ]                           # Final state (free final longitude)

#Tmax = cTmax * 60.; tf = 15.2055; p0 = -[ .361266, 22.2412, 7.87736, 0, 0, -5.90802 ]
#Tmax = cTmax * 6.0; tf = 1.320e2; p0 = -[ -4.743728539366440e+00, -7.171314869854240e+01, -2.750468309804530e+00, 4.505679923365745e+01, -3.026794475592510e+00, 2.248091067047670e+00 ]
Tmax = cTmax * 0.7; tf = 1.210e3; p0 = -[ -2.215319700438820e+01, -4.347109477345140e+01, 9.613188807286992e-01, 3.181800985503019e+02, -2.307236094862410e+00, -5.797863110671591e-01 ]
#Tmax = cTmax * 0.14; tf = 6.080e3; p0 = -[ -1.234155379067110e+02, -6.207170881591489e+02, 5.742554220129187e-01, 1.629324243017332e+03, -2.373935935351530e+00, -2.854066853269850e-01 ]

p0 = p0 / norm(p0) # Normalization |p0|=1 for free final time
ξ = [ tf ; p0 ]; # initial guess

## Hamiltonian flow and shooting function

function h(t, x, p)
    pa = x[1]
    ex = x[2]
    ey = x[3]
    hx = x[4]
    hy = x[5]
    lg = x[6]

    pdm = sqrt(pa/μ)
    cl = cos(lg)
    sl = sin(lg)
    w = 1.0 + ex*cl + ey*sl
    pdmw = pdm / w
    zz = hx*sl - hy*cl
    uh = (1 + hx^2 + hy^2) / 2

    f06 = w^2 / (pa*pdm)
    h0  = p[6] * f06

    f12 = pdm *   sl
    f13 = pdm * (-cl)
    h1  = p[2]*f12 + p[3]*f13

    f21 = pdm * 2.0 * pa / w
    f22 = pdm * (cl + (ex + cl) / w)
    f23 = pdm * (sl + (ey + sl) / w)
    h2  = p[1]*f21 + p[2]*f22 + p[3]*f23

    f32 = pdmw * (-zz*ey)
    f33 = pdmw *   zz*ex
    f34 = pdmw * uh * cl
    f35 = pdmw * uh * sl
    f36 = pdmw * zz
    h3  = p[2]*f32 + p[3]*f33 + p[4]*f34 + p[5]*f35 + p[6]*f36

    mass = mass0 - β*Tmax*t

    ψ = sqrt(h1^2 + h2^2 + h3^2)

    r = -1 + h0 + (Tmax/mass) * ψ
    return r
end

f = Flow(h)

function shoot(tf, p0)
    xf, pf = f(t0, x0, p0, tf)
    s = zeros(eltype(tf), 7)
    s[1:5] = xf[1:5] - xf_fixed
    s[6] = pf[6]
    s[7] = p0[1]^2 + p0[2]^2 + p0[3]^2 + p0[4]^2 + p0[5]^2 + p0[6]^2 - 1
    return s
end

## Solve
foo(ξ) = shoot(ξ[1], ξ[2:end])
jfoo(ξ) = jac(foo, ξ)
foo!(s, ξ) = ( s[:] = foo(ξ); nothing )
jfoo!(js, ξ) = ( js[:] = jfoo(ξ); nothing )

#nl_sol = fsolve(foo!,        ξ, show_trace=true); println(nl_sol)
nl_sol = fsolve(foo!, jfoo!, ξ, show_trace=true); println(nl_sol)

if nl_sol.converged
    tf = nl_sol.x[1]; p0 = nl_sol.x[2:end]
else
    error("Not converged")
end

ode_sol = f((t0, tf), x0, p0)
t  = ode_sol.t; N = size(t, 1)
P  = ode_sol[1, :]
ex = ode_sol[2, :]
ey = ode_sol[3, :]
hx = ode_sol[4, :]
hy = ode_sol[5, :]
L  = ode_sol[6, :]
cL = cos.(L)
sL = sin.(L)
W  = @. 1 + ex*cL + ey*sL
Z  = @. hx*sL - hy*cL
C  = @. 1 + hx^2 + hy^2
q1 = @. P *( (1 + hx^2 - hy^2)*cL + 2*hx*hy*sL ) / (C*W)
q2 = @. P *( (1 - hx^2 + hy^2)*sL + 2*hx*hy*cL ) / (C*W)
q3 = @. 2*P*Z / (C*W)

plt1 = plot3d(xlim = (-60, 60), ylim = (-60, 60), zlim = (-5, 5), title = "Orbit transfer", legend=false)
plot3d!(plt1, q1, q2, q3)

plt2 = plot3d(xlim = (-60, 60), ylim = (-60, 60), zlim = (-5, 5), title = "Orbit transfer", legend=false)
@gif for i = 1:N
    push!(plt2, q1[i], q2[i], q3[i])
end every N ÷ 100