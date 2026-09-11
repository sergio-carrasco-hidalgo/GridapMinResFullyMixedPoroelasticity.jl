module BiotPrecTest

using Gridap
import Gridap: ∇
using Gridap.Algebra
using Gridap.Geometry: simplexify
using Printf
using Test
using Krylov
using LinearAlgebra
using LinearOperators
using DataFrames

function extract_component(component)
    return x -> x[component]
end

function extract_row2d(row)
    return x -> VectorValue(x[1,row],x[2,row])
end

comp1=extract_component(1)
comp2=extract_component(2)
row1=extract_row2d(1)
row2=extract_row2d(2)


# ---------------------------------------------------------------------------
# A Gridap direct-solver factorization wrapped as a LinearOperators.jl operator
# so it can be used as a (block) preconditioner inside Krylov.jl solvers.
# ---------------------------------------------------------------------------

struct GridapLinearSolverPreconditioner{A,B}
    numerical_setup::A
    matrix::B
    function GridapLinearSolverPreconditioner(matrix; ls=LUSolver())
        ss = symbolic_setup(ls, matrix)
        ns = numerical_setup(ss, matrix)
        new{typeof(ns),typeof(matrix)}(ns, matrix)
    end
end

# Solve M*z = r using the wrapped numerical factorization.
function LinearAlgebra.mul!(z::AbstractVector{T},
                             M::GridapLinearSolverPreconditioner,
                             r::AbstractVector{T}) where {T}
    Gridap.solve!(z, M.numerical_setup, r)
end

function LinearOperators.LinearOperator(solver::GridapLinearSolverPreconditioner)
    function apply(res, x, κ, β)
        if β ≈ 0.0
            mul!(res, solver, x)
        else
            @assert κ ≈ 1.0
            tmp = copy(res)
            mul!(tmp, solver, x)
            res .= β .* res .+ κ .* tmp
        end
    end
    LinearOperator(Float64, size(solver.matrix, 1), size(solver.matrix, 2), true, true, apply)
end

# ---------------------------------------------------------------------------
# Manufactured solution and geometry
# ---------------------------------------------------------------------------

# Uniform triangular mesh of the unit square, refined 2^nk times per direction.
function generate_model2d(nk)
    domain = (0, 1, 0, 1)
    n = 2^nk
    partition = (n, n)
    CartesianDiscreteModel(domain, partition) |> simplexify
end

# Split the unit-square boundary into "Gamma_p" (top, right, and their corners),
# carrying essential data for p, and "Gamma_u" (bottom, left, bottom-left corner),
# carrying essential data for ζ.
function setup_model_labels_unit_square!(model)
    labels = get_face_labeling(model)
    add_tag!(labels, "Gamma_D", [6, 2, 3, 4, 8])  # top, right, 3 corners
    add_tag!(labels, "Gamma_N", [1, 2, 3, 5, 7])  # bottom, left, bottom-left corner
end

# Material parameters
const d  = 2
const E  = 1.0e2
const ν  = 0.49
const λ  = (E*ν)/((1+ν)*(1-2*ν))
const μ  = E/(2*(1+ν))
const α  = 1.0      # Biot-Willis coefficient
const s0 = 1.0e-6   # storativity
const κ  = 1.0e-5   # permeability over fluid viscosity
const β  = α/(2μ + d*λ)
const w0 = s0 + d*α*β

tr(τ1,τ2)  = comp1∘τ1 + comp2∘τ2
skw(τ1,τ2) = comp2∘τ1 - comp1∘τ2          
Cinv(σ1,σ2,τ1,τ2) = (1/(2μ))*(σ1⋅τ1 + σ2⋅τ2) - (λ/(2μ*(2μ+d*λ)))*tr(σ1,σ2)*tr(τ1,τ2)

nk = 2
model = generate_model2d(nk)
setup_model_labels_unit_square!(model)

# ---------------------------------------------------------------------------
# Manufactured solution
#   σ = 2μ ε(u) + λ(∇⋅u)I - α p I,   z = -κ ∇p,   γ = skew rotation
# The γ scaling matches the convention skw(τ) = τ₁₂ - τ₂₁ used above.
# ---------------------------------------------------------------------------

const I2 = TensorValue(1.0, 0.0, 0.0, 1.0)

p_ex(x) = cos(2π * x[1]) * cos(2π * x[2])
u_ex(x) = VectorValue(sin(π*x[1])*sin(π*x[2]), cos(π*x[1])*cos(π*x[2]))
gradu_ex(x) = ∇(u_ex)(x)
ε_ex(x) = 0.5*(gradu_ex(x) + transpose(gradu_ex(x)))
γ_ex(x) = 0.5*(comp2∘row1∘gradu_ex(x) - comp1∘row2∘gradu_ex(x))

σ_ex(x)  = 2μ*ε_ex(x) + (λ*(∇⋅u_ex)(x) - α*p_ex(x))*I2
σ1_ex(x) = row1(σ_ex(x))
σ2_ex(x) = row2(σ_ex(x))

z_ex(x) = -κ*∇(p_ex)(x)

# Volume data:  ∇⋅σ = f  and  s0p + α(∇⋅u) + ∇⋅z = m
u1_ex(x) = comp1(u_ex(x))
u2_ex(x) = comp2(u_ex(x))
f_ex(x)  = VectorValue((∇⋅σ1_ex)(x), (∇⋅σ2_ex)(x))
m_ex(x)  = s0*p_ex(x) + α*(∇⋅u_ex)(x) + (∇⋅z_ex)(x)

reffe_bdm = ReferenceFE(bdm, Float64, 1)
reffe_P0v = ReferenceFE(lagrangian, VectorValue{2,Float64}, 0)
reffe_P0  = ReferenceFE(lagrangian, Float64, 0)

# ---- test ----
Σ1_ = TestFESpace(model, reffe_bdm, dirichlet_tags="Gamma_N", conformity=:HDiv)  # row 1 sigma
Σ2_ = TestFESpace(model, reffe_bdm, dirichlet_tags="Gamma_N", conformity=:HDiv)  # row 2 sigma
P_  = TestFESpace(model, reffe_P0,  conformity=:L2)
U_  = TestFESpace(model, reffe_P0v, conformity=:L2)
G_  = TestFESpace(model, reffe_P0,  conformity=:L2)   # dof of the skw 
Z_  = TestFESpace(model, reffe_bdm, dirichlet_tags="Gamma_N", conformity=:HDiv)

# ---- trial ----
Σ1 = TrialFESpace(Σ1_, σ1_ex)
Σ2 = TrialFESpace(Σ2_, σ2_ex)
P  = TrialFESpace(P_)      
U  = TrialFESpace(U_)
G  = TrialFESpace(G_)
Z  = TrialFESpace(Z_, z_ex)

Xh = MultiFieldFESpace([Σ1, Σ2, P, U, G, Z])
Yh = MultiFieldFESpace([Σ1_, Σ2_, P_, U_, G_, Z_])   # (σ1,σ2,p,u,γ,z)

degree = 4

Ω  = Triangulation(model)
dΩ = Measure(Ω, degree)

# Essential data lives on Gamma_N (σ⋅n and z⋅n, imposed through the trial
# spaces above); u and p enter naturally through Gamma_D below.
Γ_D  = BoundaryTriangulation(model, tags="Gamma_D")
dΓ_D = Measure(Γ_D, degree)
n_D  = get_normal_vector(Γ_D)

a((σ1,σ2,p),(τ1,τ2,q)) = ∫( Cinv(σ1,σ2,τ1,τ2) + β*p*tr(τ1,τ2) + β*q*tr(σ1,σ2) + w0*p*q )dΩ
b((τ1,τ2,q),(v,γ,w)) = ∫( (comp1∘v)*(∇⋅τ1) + (comp2∘v)*(∇⋅τ2) + skw(τ1,τ2)*γ + q*(∇⋅w) )dΩ
c((u,γ,z), (v,η,w)) = ∫( (1/κ)*(z⋅w) )dΩ

lhs((σ1,σ2,p,u,γ,z),(τ1,τ2,q,v,η,w)) =
    a((σ1,σ2,p),(τ1,τ2,q)) + b((τ1,τ2,q),(u,γ,z)) + b((σ1,σ2,p),(v,η,w)) - c((u,γ,z),(v,η,w))

rhs((τ1,τ2,q,v,η,w)) =
    ∫( u1_ex*(τ1⋅n_D) + u2_ex*(τ2⋅n_D) + p_ex*(w⋅n_D) )dΓ_D +
    ∫( f_ex⋅v + m_ex*q )dΩ

op = AffineFEOperator(lhs, rhs, Xh, Yh)

@info "affine operator assembled"

Σ_test  = MultiFieldFESpace([Σ1_, Σ2_])
Σ_trial = MultiFieldFESpace([Σ1,  Σ2 ])

# Aσ_  = assemble_matrix(rσ,   Σ_trial, Σ_test)
# Apa_ = assemble_matrix(rp_a, P,  P_)
# Apb_ = assemble_matrix(rp_b, P,  P_)
# Au_  = assemble_matrix(ru,   U,  U_)
# Ag_  = assemble_matrix(rγ,   G,  G_)
# Az_  = assemble_matrix(rz,   Z,  Z_)

# DOF ranges of ALL BLOCKS within the global system, used to place
# the preconditioner's block inverses into the dense Pinv matrix below.

Ns = [num_free_dofs(U) for U in op.trial]
Np = zeros(Int, length(Ns) + 1)
Np[1] = 1
for i in 2:length(Ns)+1
    Np[i] = Np[i-1] + Ns[i-1]
end
range_σ = Np[1]:Np[3]-1   # σ block (adding σ1 and σ2)
range_p = Np[3]:Np[4]-1   # p block
range_u = Np[4]:Np[5]-1   # u block
range_γ = Np[5]:Np[6]-1   # γ block
range_z = Np[6]:Np[7]-1   # z block

N = num_free_dofs(op.trial)
Pinv = zeros(N, N)
Adense = Array(op.op.matrix)

@info "dense A computed"

"""
Preconditioner forms:

block σ = (Cinv - 1/2μ∇∇⋅)^{-1}
block p = ((1 + dαβ + s0)Id)^{-1} + ((s0+dαβ)Id - κΔ)^{-1}
block u = block γ = 1/2\mu Id
block z = (1/κ Id - ∇∇⋅)^{-1}
"""

end