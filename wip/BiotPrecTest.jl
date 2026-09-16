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


# tr(τ1,τ2)  = comp1∘τ1 + comp2∘τ2
# skw(τ1,τ2) = comp2∘τ1 - comp1∘τ2          
# Cinv(σ1,σ2,τ1,τ2) = (1/(2μ))*(σ1⋅τ1 + σ2⋅τ2) - (λ/(2μ*(2μ+d*λ)))*tr(σ1,σ2)*tr(τ1,τ2)

nk = 2
model = generate_model2d(nk)
setup_model_labels_unit_square!(model)

const I2 = TensorValue(1.0, 0.0, 0.0, 1.0)

p_ex(x) = cos(2π * x[1]) * cos(2π * x[2])
u_ex(x) = VectorValue(sin(π*x[1])*sin(π*x[2]), cos(π*x[1])*cos(π*x[2]))

function assemble_biot_precond(model, p_ex, u_ex, params; nk=2, degree=4)

    d  = params.d
    μ  = params.μ
    λ  = params.λ
    α  = params.α
    s0 = params.s0
    κ  = params.κ

    β  = α/(2μ + d*λ)
    w0 = s0 + d*α*β

    trt(τ1,τ2) = (comp1∘τ1) + (comp2∘τ2)
    skw(τ1,τ2) = (comp2∘τ1) - (comp1∘τ2)
    Cinv(σ1,σ2,τ1,τ2) = (1/(2μ))*(σ1⋅τ1 + σ2⋅τ2) - (λ/(2μ*(2μ + d*λ)))*trt(σ1,σ2)*trt(τ1,τ2)


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

    a((σ1,σ2,p),(τ1,τ2,q)) = ∫( Cinv(σ1,σ2,τ1,τ2) + β*p*trt(τ1,τ2) + β*q*trt(σ1,σ2) + w0*p*q )dΩ
    b((τ1,τ2,q),(v,γ,w)) = ∫( (comp1∘v)*(∇⋅τ1) + (comp2∘v)*(∇⋅τ2) + skw(τ1,τ2)*γ + q*(∇⋅w) )dΩ
    c((u,γ,z), (v,η,w)) = ∫( (1/κ)*(z⋅w) )dΩ

    lhs((σ1,σ2,p,u,γ,z),(τ1,τ2,q,v,η,w)) =
        a((σ1,σ2,p),(τ1,τ2,q)) + b((τ1,τ2,q),(u,γ,z)) + b((σ1,σ2,p),(v,η,w)) - c((u,γ,z),(v,η,w))

    rhs((τ1,τ2,q,v,η,w)) =
        ∫( u1_ex*(τ1⋅n_D) + u2_ex*(τ2⋅n_D) + p_ex*(w⋅n_D) )dΓ_D +
        ∫( f_ex⋅v + m_ex*q )dΩ

    op = AffineFEOperator(lhs, rhs, Xh, Yh)

    #@info "affine operator assembled"

    Σ_test  = MultiFieldFESpace([Σ1_, Σ2_])
    Σ_trial = MultiFieldFESpace([Σ1,  Σ2 ])


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

    #@info "dense A computed"

    """
    Preconditioner forms:

    block σ = (Cinv - 1/2μ∇∇⋅)^{-1}
    block p = ((1 + dαβ + s0)Id)^{-1} + ((s0+dαβ)Id - κΔ)^{-1}
    block u = block γ = 1/2  μ Id
    block z = (1/κ Id - ∇∇⋅)^{-1}
    """

    # for discrete laplacian Δh 

    Λ  = SkeletonTriangulation(model)
    dΛ = Measure(Λ, degree)
    h_e    = CellField(get_array(∫(1)*dΛ),   Λ)
    h_e_ΓD = CellField(get_array(∫(1)*dΓ_D), Γ_D)

    # Riesz operators

    a_11((σ1,σ2),(τ1,τ2)) = ∫( Cinv(σ1,σ2,τ1,τ2) +  (1/(2*μ))*((∇⋅σ1)*(∇⋅τ1) +(∇⋅σ2)*(∇⋅τ2)) )dΩ
    a_22a(p,q) = ∫( (1.0 + w0) * p * q )dΩ
    a_22b(p,q) = ∫( w0*p*q )dΩ +
                ∫(κ*(∇(p)⋅ ∇(q)))dΩ +
                ∫((κ/ h_e) * jump(p) * jump(q))dΛ +
                ∫((κ/ h_e_ΓD) * p * q)dΓ_D

    a_21a(u,v) = ∫(2*μ*(u⋅v))dΩ
    a_21b(γ,η) = ∫(2*μ*(γ*η))dΩ

    a_33(z,w) = ∫((1/κ)*z⋅w + ((∇⋅z) * (∇⋅w)))dΩ 

    A_σ  = assemble_matrix(a_11,  Σ_trial, Σ_test)
    A_pa = assemble_matrix(a_22a, P, P_)
    A_pb = assemble_matrix(a_22b, P, P_)
    A_u  = assemble_matrix(a_21a,  U, U_)
    A_γ  = assemble_matrix(a_21b,  G, G_)
    A_z  = assemble_matrix(a_33,  Z, Z_)

    #@info "Assembly done"

    B_σ = LinearOperator(GridapLinearSolverPreconditioner(A_σ))
    B_p = LinearOperator(GridapLinearSolverPreconditioner(A_pa)) +
        LinearOperator(GridapLinearSolverPreconditioner(A_pb))   
    B_u = LinearOperator(GridapLinearSolverPreconditioner(A_u))
    B_γ = LinearOperator(GridapLinearSolverPreconditioner(A_γ))
    B_z = LinearOperator(GridapLinearSolverPreconditioner(A_z))

    #@info "Block operators assembled"

    riesz = BlockDiagonalOperator(B_σ, B_p, B_u, B_γ, B_z)

    #@info "Riesz operator assembled"

    Pinv[range_σ, range_σ] = inv(Array(A_σ))
    Pinv[range_p, range_p] = inv(Array(A_pa)) + inv(Array(A_pb))
    Pinv[range_u, range_u] = inv(Array(A_u))
    Pinv[range_γ, range_γ] = inv(Array(A_γ))
    Pinv[range_z, range_z] = inv(Array(A_z))

    evals = eigvals(Pinv * Adense)

    return op, riesz, evals

end
    Base.@kwdef struct BiotParams
        d::Int      = 2
        μ::Float64  = 1.0
        λ::Float64  = 1.0
        α::Float64  = 1.0
        s0::Float64 = 1.0e-6
        κ::Float64  = 1.0e-5
    end

    table = DataFrame(nk=Int[], λ=Float64[], μ=Float64[], κ=Float64[], s0=Float64[], α=Float64[],
                     niter=Int[], solved=Bool[], cond=Float64[])

    for nk in (4)
        model = generate_model2d(nk)
        setup_model_labels_unit_square!(model)

        for λ in (1.0e-2, 1.0, 1e4, 1e8), κ in (1e-8, 1e-5, 1e-3), s0 in (1e-9, 1e-3), α in (1e-4, 1e-2, 1.0), μ in (1.0e-2, 1.0, 1.0e2)
            params = BiotParams(λ=λ, κ=κ, s0=s0, α=α, μ=μ )
            println("\n--- nk=$nk λ=$λ μ=$μ κ=$κ s0=$s0 α=$α---")

            op, riesz, evals = assemble_biot_precond(model, p_ex, u_ex, params; nk=nk)

            x, hist = minres(op.op.matrix, op.op.vector; M=riesz, itmax=2000, atol=1e-10, rtol=1e-10)

            cnd = maximum(abs, evals) / minimum(abs, evals)
            @printf("MINRES converged in %d iterations\n", hist.niter)
            @printf("condition number of the preconditioned system: %1.3e\n", cnd)

            push!(table, (nk, λ, μ, κ, s0, α, hist.niter, hist.solved, cnd))
        end # sweep params
    end #for nk

    @show table 

end #module