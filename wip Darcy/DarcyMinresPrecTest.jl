module DarcyMinresPrecTest

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

# Split the unit-square boundary
function setup_model_labels_unit_square!(model)
    labels = get_face_labeling(model)
    add_tag!(labels, "Gamma_D", [6, 2, 3, 4, 8])  # top, right, 3 corners
    add_tag!(labels, "Gamma_N", [1, 2, 3, 5, 7])  # bottom, left, bottom-left corner
end

# Material parameters
# const d  = 2
# const s0 = 1.0e-6   # storativity
# const κ  = 1.0e-5   # permeability over fluid viscosity


# nk = 2
# model = generate_model2d(nk)
# setup_model_labels_unit_square!(model)

const I2 = TensorValue(1.0, 0.0, 0.0, 1.0)

p_ex(x) = cos(2π * x[1]) * cos(2π * x[2])

function assemble_darcy_precond(model, p_ex, params; nk=2, degree=4)

    s0 = params.s0
    κ  = params.κ

    z_ex(x) = -κ*∇(p_ex)(x)

    # Volume data: s0*p + ∇⋅z = m
    m_ex(x)  = s0*p_ex(x) + (∇⋅z_ex)(x) 

    # ---- reference FEs ----
    reffe_bdm = ReferenceFE(bdm, Float64, 1)
    reffe_P0  = ReferenceFE(lagrangian, Float64, 0)

    # for larger (initial) test spaces for MinRes formulation
    reffe_bdm_1 = ReferenceFE(bdm, Float64, 2) 
    reffe_P0_1  = ReferenceFE(lagrangian, Float64, 1)

    # ---- test ----
    P_  = TestFESpace(model, reffe_P0,  conformity=:L2)
    Z_  = TestFESpace(model, reffe_bdm, dirichlet_tags="Gamma_N", conformity=:HDiv)

    rP_ = TestFESpace(model, reffe_P0_1,  conformity=:L2)
    rZ_  = TestFESpace(model, reffe_bdm_1, dirichlet_tags="Gamma_N", conformity=:HDiv)

    # ---- trial ----
    P  = TrialFESpace(P_)
    Z  = TrialFESpace(Z_, z_ex)

    # spaces for Riesz representatives
    rP  = TrialFESpace(rP_)
    rZ  = TrialFESpace(rZ_)

    Xh = MultiFieldFESpace([rP, rZ, P, Z])
    Yh = MultiFieldFESpace([rP_, rZ_, P_, Z_])  

    Ω  = Triangulation(model)
    dΩ = Measure(Ω, degree)

    # Essential data lives on Gamma_N (z⋅n, imposed through the trial
    # spaces above); p enter naturally through Gamma_D below.
    Γ_D  = BoundaryTriangulation(model, tags="Gamma_D")
    dΓ_D = Measure(Γ_D, degree)
    n_D  = get_normal_vector(Γ_D)
    
    Λ  = SkeletonTriangulation(model)
    dΛ = Measure(Λ, degree)
    h_e    = CellField(get_array(∫(1)*dΛ),   Λ)
    h_e_ΓD = CellField(get_array(∫(1)*dΓ_D), Γ_D)

  a(p,q) = ∫(s0*(p*q))dΩ
  b(q,w) = ∫( q*(∇⋅w) )dΩ 
  c(z,w) = ∫((1/κ)*(z⋅w))dΩ

  # ---- inner products ----
  d(z,w) = ∫((1/κ)*(z⋅w) + (∇⋅z)*(∇⋅w))dΩ 

  ##### Question: What even is the inner product here? Is there a way to implement an inner product based on the inf norm? 
  e(p,q) = ∫(p * q )dΩ + ∫(κ*(∇(p)⋅ ∇(q)))dΩ + ∫((κ/ h_e) * jump(p) * jump(q))dΛ + ∫((κ/ h_e_ΓD) * p * q)dΓ_D
  # ----

  lhs((εp,εz,p,z),(q,w,φp,φz)) =  a(p,q) + b(p,w) + b(q,z) - c(z,w) + a(φp,εp) + b(φp,εz) + b(εp,φz) - c(φz,εz) + e(εp,q) + d(εz,w) 
  rhs((q,w,φp,φz)) =  ∫(  p_ex*(w⋅n_D) )dΓ_D + ∫( m_ex*q )dΩ

    op = AffineFEOperator(lhs, rhs, Xh, Yh)

    #@info "affine operator assembled"

    Ns = [num_free_dofs(U) for U in op.trial]
    Np = zeros(Int, length(Ns) + 1)
    Np[1] = 1
    for i in 2:length(Ns)+1
        Np[i] = Np[i-1] + Ns[i-1]
    end

    range_εp = Np[1]:Np[2]-1
    range_εz = Np[2]:Np[3]-1
    range_p = Np[3]:Np[4]-1
    range_z = Np[4]:Np[5]-1


    N = num_free_dofs(op.trial)
    Pinv = zeros(N, N)
    Adense = Array(op.op.matrix)

    #@info "dense A computed"

    """ 
    Preconditioner forms:
    block εp = ((1 + s0)Id)^{-1} + (s0 Id - κΔ)^{-1}
    block εz = (1/κ Id - ∇∇⋅)^{-1}
    block p = ((1 + s0)Id)^{-1} + (s0 Id - κΔ)^{-1}
    block z = (1/κ Id - ∇∇⋅)^{-1}
    """

    # for discrete laplacian Δh 

    # Λ  = SkeletonTriangulation(model)
    # dΛ = Measure(Λ, degree)
    # h_e    = CellField(get_array(∫(1)*dΛ),   Λ)
    # h_e_ΓD = CellField(get_array(∫(1)*dΓ_D), Γ_D)

    # Riesz operators

    a_22a(p,q) = ∫( (1.0 + s0) * p * q )dΩ
    a_22b(p,q) = ∫( s0*p*q )dΩ +
                ∫(κ*(∇(p)⋅ ∇(q)))dΩ +
                ∫((κ/ h_e) * jump(p) * jump(q))dΛ +
                ∫((κ/ h_e_ΓD) * p * q)dΓ_D

    a_33(z,w) = ∫((1/κ)*z⋅w + ((∇⋅z) * (∇⋅w)))dΩ 

    A_εpa = assemble_matrix(a_22a, rP, rP_)
    A_εpb = assemble_matrix(a_22b, rP, rP_)
    A_εz  = assemble_matrix(a_33, rZ, rZ_)
    A_pa = assemble_matrix(a_22a, P, P_)
    A_pb = assemble_matrix(a_22b, P, P_)
    A_z  = assemble_matrix(a_33,  Z, Z_)

    #@info "Assembly done"

    B_εp = LinearOperator(GridapLinearSolverPreconditioner(A_εpa)) +
        LinearOperator(GridapLinearSolverPreconditioner(A_εpb))   
    B_εz = LinearOperator(GridapLinearSolverPreconditioner(A_εz))
    B_p = LinearOperator(GridapLinearSolverPreconditioner(A_pa)) +
        LinearOperator(GridapLinearSolverPreconditioner(A_pb))   
    B_z = LinearOperator(GridapLinearSolverPreconditioner(A_z))

    #@info "Block operators assembled"

    riesz = BlockDiagonalOperator(B_εp, B_εz, B_p, B_z)

    #@info "Riesz operator assembled"

    Pinv[range_εp, range_εp] = inv(Array(A_εpa)) + inv(Array(A_εpb))
    Pinv[range_εz, range_εz] = inv(Array(A_εz))
    Pinv[range_p, range_p] = inv(Array(A_pa)) + inv(Array(A_pb))
    Pinv[range_z, range_z] = inv(Array(A_z))

    evals = eigvals(Pinv * Adense)

    return op, riesz, evals

end
    Base.@kwdef struct DarcyParams
        d::Int      = 2
        s0::Float64 = 1.0e-6
        κ::Float64  = 1.0e-5
    end

    table = DataFrame(nk=Int[], κ=Float64[], s0=Float64[],
                     niter=Int[], solved=Bool[], cond=Float64[])

    for nk in (2)
        model = generate_model2d(nk)
        setup_model_labels_unit_square!(model)

        for κ in (1e-9, 1e-6, 1e-3), s0 in (1e-9, 1e-6, 1e-3)
            params = DarcyParams(κ=κ, s0=s0)
            println("\n--- nk=$nk κ=$κ s0=$s0 ---")

            op, riesz, evals = assemble_darcy_precond(model, p_ex, params; nk=nk)

            x, hist = minres(op.op.matrix, op.op.vector; M=riesz, itmax=2000, atol=1e-10, rtol=1e-10)

            cnd = maximum(abs, evals) / minimum(abs, evals)
            @printf("MINRES converged in %d iterations\n", hist.niter)
            @printf("condition number of the preconditioned system: %1.3e\n", cnd)

            push!(table, (nk, κ, s0, hist.niter, hist.solved, cnd))
        end # sweep params
    end #for nk

    @show table 

end #module