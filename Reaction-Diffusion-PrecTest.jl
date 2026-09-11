# Boon-type block-diagonal preconditioners for a mixed (H(div) x L2) discretization
# of the perturbed reaction-diffusion / Darcy system, tested via MINRES against a
# manufactured solution over a range of (κ, s₀) parameters and mesh refinements.
module PerturbedDarcy

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
# using AlgebraicMultigrid

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

# Exact scalar field p used to manufacture the RHS and boundary data.
p_ex(x) = cos(2π * x[1]) * cos(2π * x[2])

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
    add_tag!(labels, "Gamma_p", [6, 2, 3, 4, 8])  # top, right, 3 corners
    add_tag!(labels, "Gamma_u", [1, 2, 3, 5, 7])  # bottom, left, bottom-left corner
end

# ---------------------------------------------------------------------------
# Assemble the mixed reaction-diffusion system and its Boon-type block-diagonal
# preconditioner (variant :B1 or :B2). Returns the FE operator, the
# preconditioner as a LinearOperators.jl block-diagonal operator (for use in
# Krylov.jl), and the eigenvalues of the (dense) preconditioned system.
# ---------------------------------------------------------------------------
function assemble_reaction_diffusion_precond(model, p_ex; k, κ, s₀, prec_variant)
    @assert prec_variant in (:B1, :B2)

    # Manufactured data for the perturbed Darcy system:
    #   ζ = κ ∇p,   s₀ p - ∇⋅ζ = m,   (1/κ) ζ - ∇p = h 
    ζ_ex(x) = κ * ∇(p_ex)(x)
    m_ex(x) = s₀ * p_ex(x) - (∇ ⋅ ζ_ex)(x)
    h_ex(x) = (1.0 / κ) * ζ_ex(x) - ∇(p_ex)(x)

    # FE spaces: Raviart-Thomas for ζ, discontinuous Lagrange for p.
    reffe_ζ = ReferenceFE(raviart_thomas, Float64, k)
    reffe_p = ReferenceFE(lagrangian, Float64, k)

    Wh_ = TestFESpace(model, reffe_ζ, dirichlet_tags="Gamma_p", conformity=:HDiv)
    qh_ = TestFESpace(model, reffe_p, conformity=:L2)

    Wh = TrialFESpace(Wh_, ζ_ex)
    qh = TrialFESpace(qh_)

    Yh = MultiFieldFESpace([Wh_, qh_])
    Xh = MultiFieldFESpace([Wh, qh])

    # Integration measures.
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2 * (k + 2))

    Γu = BoundaryTriangulation(model, tags="Gamma_u")
    dΓu = Measure(Γu, 2 * (k + 2) - 1)
    n_Γu = get_normal_vector(Γu)

    # Reserved for a Γp boundary term (see the commented-out e(p,ξ) term below).
    Γp = BoundaryTriangulation(model, tags="Gamma_p")
    dΓp = Measure(Γp, 2 * (k + 2) - 1)
    n_Γp = get_normal_vector(Γp)

    Λu = SkeletonTriangulation(model)
    dΛu = Measure(Λu, 2 * (k + 2) - 1)
    h_eu = CellField(get_array(∫(1) * dΛu), Λu)     # interior facet size, for jump stabilization
    h_e_Γu = CellField(get_array(∫(1) * dΓu), Γu)   # boundary facet size on Gamma_u

    # Bilinear/linear forms of the mixed problem.
    a(ζ, ξ) = ∫((1.0 / κ) * (ζ ⋅ ξ)) * dΩ
    b(ξ, q) = ∫((∇ ⋅ ξ) * q) * dΩ
    c(p, q) = ∫(s₀ * p * q) * dΩ

    H(ξ) = ∫(p_ex * (ξ ⋅ n_Γu)) * dΓu + ∫(h_ex ⋅ ξ) * dΩ
    M(q) = ∫(-(m_ex * q)) * dΩ

    # + e(p,ξ) was a candidate extra Γp boundary term, currently disabled.
    lhs((ζ, p), (ξ, q)) = a(ζ, ξ) + b(ξ, p) + b(ζ, q) - c(p, q)
    rhs((ξ, q)) = H(ξ) + M(q)

    op = AffineFEOperator(lhs, rhs, Xh, Yh)

    # DOF ranges of the ζ- and p-blocks within the global system, used to place
    # the preconditioner's block inverses into the dense Pinv matrix below.
    Ns = [num_free_dofs(U) for U in op.trial]
    Np = zeros(Int, length(Ns) + 1)
    Np[1] = 1
    for i in 2:length(Ns)+1
        Np[i] = Np[i-1] + Ns[i-1]
    end
    range1 = Np[1]:Np[2]-1   # ζ block
    range2 = Np[2]:Np[3]-1   # p block

    N = num_free_dofs(op.trial)
    Pinv = zeros(N, N)
    Adense = Array(op.op.matrix)

    # Boon-type preconditioner forms. B1 splits the p-block into two pieces
    # (reaction mass, and diffusion + interior/boundary jump stabilization)
    # inverted separately; B2 folds both pieces into a single p-block form.
    #
    # NOTE: each branch below MUST use distinctly-named local functions
    # (a11_B1 vs a11_B2, etc.) rather than a shared name like `a11` reused
    # across both branches. Reusing one name across branches
    # makes the two branches silently overwrite each other's method in the
    # global method table.
    if prec_variant == :B1
        a11_B1(ζ, ξ) = ∫((1 / κ) * (ζ ⋅ ξ) + (∇ ⋅ ζ) * (∇ ⋅ ξ)) * dΩ

        a22a_B1(p, q) = ∫((1.0 + s₀) * p * q) * dΩ
        a22b_B1(p, q) = ∫(s₀ * p * q) * dΩ +
                        ∫(κ * (∇(p)) ⋅ ∇(q)) * dΩ +
                        ∫((κ / h_eu) * jump(p) * jump(q)) * dΛu +
                        ∫((κ / h_e_Γu) * p * q) * dΓu

        A11_ = assemble_matrix(a11_B1, Wh, Wh_)
        A22a_ = assemble_matrix(a22a_B1, qh, qh_)
        A22b_ = assemble_matrix(a22b_B1, qh, qh_)

        A22 = LinearOperator(GridapLinearSolverPreconditioner(A22a_)) +
              LinearOperator(GridapLinearSolverPreconditioner(A22b_))

        Pinv[range2, range2] = inv(Array(A22a_)) + inv(Array(A22b_))
    else # :B2
        a11_B2(ζ, ξ) = ∫((1.0 / κ) * (ζ ⋅ ξ) + (1.0 / κ) * (∇ ⋅ ζ) * (∇ ⋅ ξ)) * dΩ

        a22_B2(p, q) = ∫((s₀ + 1.0 / κ) * p * q) * dΩ +
                       ∫(κ * (∇(p)) ⋅ ∇(q)) * dΩ +
                       ∫((κ / h_eu) * jump(p) * jump(q)) * dΛu +
                       ∫((κ / h_e_Γu) * p * q) * dΓu

        A11_ = assemble_matrix(a11_B2, Wh, Wh_)
        A22_ = assemble_matrix(a22_B2, qh, qh_)

        A22 = LinearOperator(GridapLinearSolverPreconditioner(A22_))

        Pinv[range2, range2] = inv(Array(A22_))
    end

    A11 = LinearOperator(GridapLinearSolverPreconditioner(A11_))
    Pinv[range1, range1] = inv(Array(A11_))

    evals = eigvals(Pinv * Adense)

    return op, BlockDiagonalOperator(A11, A22), evals
end

# ---------------------------------------------------------------------------
# Parameter sweep: for each mesh refinement / (κ, s₀) pair, assemble the
# system and its :B1 preconditioner, solve with MINRES, and record the
# iteration count and the condition number of the preconditioned system.
# ---------------------------------------------------------------------------
table = DataFrame(nk=Int[], κ=Float64[], s₀=Float64[], niter=Int[], cond_num=Float64[])

for nk in (2, 3, 4)
    model = generate_model2d(nk)
    setup_model_labels_unit_square!(model)

    for κ in (1e-5, 1e-4, 1e-3), s₀ in (1e-9, 1e-6, 1e-3)
        println("\n--- Running for nk=$nk, κ=$κ, s₀=$s₀ ---")

        oper, riesz, evals = assemble_reaction_diffusion_precond(model, p_ex; k=0, κ=κ, s₀=s₀, prec_variant=:B1)

        A = oper.op.matrix
        rhs_vec = oper.op.vector
        x, hist = minres(A, rhs_vec, M=riesz, itmax=1000, atol=1e-8, rtol=1e-8, verbose=0)

        cond_raw = maximum(abs.(evals)) / minimum(abs.(evals))
        @printf("MINRES converged in %d iterations\n", hist.niter)
        @printf("condition number of the preconditioned system: %1.3e\n", cond_raw)

        cond_num = round(cond_raw, digits=1)
        push!(table, (nk=nk, κ=κ, s₀=s₀, niter=hist.niter, cond_num=cond_num))
    end
end

@show table

end # module