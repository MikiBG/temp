using LinearAlgebra, LinearAlgebra.BLAS, SparseArrays, HDF5

# compute A ⊕ B = A ⊗ I + I ⊗ B
function kronsum(A::SparseMatrixCSC{Float64,Int64}, B::SparseMatrixCSC{Float64,Int64})
  kron(A, sparse(1.0I, size(B))) + kron(sparse(1.0I, size(A)), B)
end

function kronsum(A::SparseMatrixCSC{Complex{Float64},Int64}, B::SparseMatrixCSC{Complex{Float64},Int64})
  kron(A, sparse((1.0 + 0.0im)I, size(B))) + kron(sparse((1.0 + 0.0im)I, size(A)), B)
end

# compute the Kronecker sum of several operators
function kronsum(A, B, more...)
  R = kronsum(A, B)
  for X in more
    R = kronsum(R, X)
  end
  R
end

# compute the Kronecker sum of the same operator A, N times
function kronsum(A::SparseMatrixCSC{Float64,Int64}, N::Int)
  R = A
  for i in 1:(N - 1)
    R = kronsum(R, A)
  end
  R
end

# compute the trace of a matrix (either flattened to a vector or in the usual 2D form)
function trace(rho)
  dim = isqrt(length(rho))
  total = rho[1]
  @inbounds for n in 2:dim
    total += rho[n*(dim + 1) - dim]
  end 
  total
end

# ppt criterion
function minPT(rho, d1, d2)
    # computes the partial transpose of a density matrix rho (should be provided in vector form).
    # the dimensions of the two subsystems must satisfy d1*d2 = sqrt(length(rho))
    # returns its minimum eigenvalue
    M = reshape(rho, d1*d2, d1*d2)
    PT = Array{eltype(rho),2}(undef, d1*d2, d1*d2)
    @inbounds for j in 1:d1
        cols = (j - 1)*d2 + 1: j*d2
        @inbounds for i in 1:d1
            rows = (i - 1)*d2 + 1: i*d2
            PT[rows, cols] = view(M, rows, cols)'
        end
    end
    eigmin(Hermitian(PT))
end

# return the matrix representation of the exact evolution superoperator corresponding 
# to a whole cycle of the protocol
function EvolutionSuperOp(mu::Float64, nu::Float64, g1::Array{Float64,1}, g2::Array{Float64,1}, k::Int)
  N1, N2 = length(g1), length(g2)
  dim1, dim2 = 2^N1, 2^N2
  Id1 = sparse(1.0I, dim1, dim1)  # Identity matrix for QD1
  Id2 = sparse(1.0I, dim2, dim2)  # Identity matrix for QD2
  Kup = kron(Id1, [1.0, 0.0], Id2)
  InUP = kron(Kup, Kup)  # Superoperator for injecting a spin up
  Kdn = kron(Id1, [0.0, 1.0], Id2)
  InDN = kron(Kdn, Kdn)  # Superoperator for injecting a spin up
  KupT = sparse(Kup')
  KdnT = sparse(Kdn')
  PT = kron(KupT, KupT) + kron(KdnT, KdnT)  # partial trace
  # Hyperfine interaction Hamiltonians. They omit the "other QD"
  Sp = sparse([0.0 1.0; 0.0 0.0])
  Sm = sparse(Sp')
  Sz = sparse([0.5 0.0; 0.0 -0.5])
  A1p = kronsum([gi1*Sp for gi1 in g1]...)
  A1z = kronsum([gi1*Sz for gi1 in g1]...)
  A1m = sparse(A1p')
  A2p = kronsum([gi2*Sp for gi2 in g2]...)
  A2z = kronsum([gi2*Sz for gi2 in g2]...)
  A2m = sparse(A2p')
  H1 = Array(kron(A1z, Sz) + (kron(A1p, Sm) + kron(A1m, Sp))*0.5)
  H2 = Array(kron(Sz, A2z) + (kron(Sm, A2p) + kron(Sp, A2m))*0.5)
  # The evolution superoperator for one full cycle is the product of the 
  # evolution superoperators for each step
  V = (PT * kron(Id1, exp(-im*nu*H2), Id1, exp(im*nu*H2))
        * kron(exp(-im*mu*H1), Id2, exp(im*mu*H1), Id2) * InUP)  
  V = (PT * kron(Id1, exp(-im*mu*H2), Id1, exp(im*mu*H2))
        * kron(exp(-im*nu*H1), Id2, exp(im*nu*H1), Id2) * InDN * V)
  V = (PT * kron(exp(-im*((mu - nu)/k)*H1), Id2, exp(im*((mu - nu)/k)*H1), Id2) * InDN)^k * V 
  (PT * kron(Id1, exp(-im*((mu - nu)/k)*H2), Id1, exp(im*((mu - nu)/k)*H2)) * InUP)^k * V 
end

# computes the evolution of an inhomogeneous system using the exact time evolution operator. the initial state is given by rho, 
# which is changed in-place after each cycle. The function returns an array containing the 
# polarizations, variance of the x-component of the total spin, norm and state purity.
function evolution!(mu::Float64, nu::Float64, k::Int, g1::Array{Float64,1}, g2::Array{Float64,1}, rho::Array{Complex{Float64},1}, measureat::Array{Int,1})
  V = Array(EvolutionSuperOp(mu, nu, g1, g2, k))
  N1, N2 = length(g1), length(g2)
  dim1, dim2 = 2^N1, 2^N2
  Id1 = sparse(1.0I, dim1, dim1)
  Id2 = sparse(1.0I, dim2, dim2)
  Id = sparse(1.0I, dim1*dim2, dim1*dim2)
  Sz = sparse([0.5 0.0; 0.0 -0.5])
  Sx = sparse([0.0 0.5; 0.5 0.0])
  Sp = sparse([0.0 1.0; 0.0 0.0])
  Az2_1 = kron(kronsum([gi1^2 * Sz for gi1 in g1]...), Id2, Id)
  Az2_2 = kron(Id1, kronsum([gi2^2 * Sz for gi2 in g2]...), Id)
  Ax1_plus_Ax2 = kron(kronsum([gi * Sx for gi in [g1; g2]]...), Id)
  Ax1_plus_Ax2_sq = Ax1_plus_Ax2^2
  evolution_data = Array{Float64, 2}(undef, length(measureat), 6)
  ind = 1
#   v0 = zeros(Complex{Float64}, dim1*dim2)
  for i in 1:measureat[end]
    rho .= gemv('N', V, rho)
    if i == measureat[ind]
      evolution_data[ind, 1] = real(trace(rho))
      evolution_data[ind, 2] = sum(abs2.(rho))
      evolution_data[ind, 3] = real(trace(Az2_1 * rho))
      evolution_data[ind, 4] = real(trace(Az2_2 * rho))
      evolution_data[ind, 5] = real(trace(Ax1_plus_Ax2_sq * rho) - trace(Ax1_plus_Ax2 * rho)^2)
      evolution_data[ind, 6] = minPT(rho, dim1, dim2)
      ind += 1
    end
  end 
  evolution_data
end

# parameters
g1 = [1.0, 0.98, 0.1]
g1 /= norm(g1)
g2 = [1.0, 0.95, 0.08, 0.02]
g2 /= norm(g2)

N1, N2 = length(g1), length(g2)
Ntot = N1 + N2
dim = 2^Ntot
mu = 0.1/sum(g1)
nu = 0.8*mu
k = 5
Ncycle = 50000000
measureat = unique(Int.(round.(10 .^ range(0, stop=log10(Ncycle), length=100))))


# initial state is a product state of polarized ensembles in opposite directions
rho = zeros(Complex{Float64}, dim^2)
stateind = Int(0b0001111) + 1 # index of the pure state. Bitwise notation: 0 = "up", 1 = "down"
rho[stateind*(dim + 1) - dim] = 1.0

evolution_data = evolution!(mu, nu, k, g1, g2, rho, measureat)

# save results
h5open("exact_evo_inhom_superhipersmall_extended.h5", "w") do file
  file["measureat"] = measureat
  file["norm"] = evolution_data[:, 1]
  file["purity"] = evolution_data[:, 2]
  file["polz1"] = evolution_data[:, 3]
  file["polz2"] = evolution_data[:, 4]
  file["varx"] = evolution_data[:, 5]
  file["minPT"] = evolution_data[:, 6]
  file["rho"] = rho
  g = g_create(file, "param")
  g["mu"] = mu
  g["nu"] = nu
  g["g1"] = g1
  g["g2"] = g2
end
