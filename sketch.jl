module Sketch

# Create type MatrixSketch
type MatrixSketch

    Y::Matrix{Float64}          # Range sketch
    W::Matrix{Float64}          # Co-range sketch

    Omega::Matrix{Float64}      # Random test matrix for the range
    Psi::Matrix{Float64}        # Random test matrix for the co-range

    r::Int64                    # Target rank

    # Initialize a sketch of the m-by-n zero matrix for parameters r, k, and l
    function MatrixSketch(m::Int64,
                          n::Int64,
                          r::Int64,
                          k::Int64,
                          l::Int64)

        # Check feasibility of sketch parameters
        if !(r<=k<=l && k<=n && l<=m)
            error("Infeasible sketch parameters r, k, and l")
        end

        # Draw and fix Gaussian normal matrices
        Omega = randn(n,k);
        Psi = randn(l,m);

        # Improve numerical stability
        Omega = qr(Omega)[1];
        Psi = qr(Psi')[1]';

        # Initialize to zero sketch
        Y = zeros(m,k);
        W = zeros(l,n);

        new(Y,W,Omega,Psi,r)

    end # function MatrixSketch

end # type MatrixSketch

# Create type ComplexMatrixSketch
type ComplexMatrixSketch

    Y::Matrix{Complex{Float64}}          # Range sketch
    W::Matrix{Complex{Float64}}          # Co-range sketch

    Omega::Matrix{Complex{Float64}}      # Random test matrix for the range
    Psi::Matrix{Complex{Float64}}        # Random test matrix for the co-range

    r::Int64                             # Target rank

    # Initialize a sketch of the m-by-n zero matrix for parameters r, k, and l
    function ComplexMatrixSketch(m::Int64,
                                 n::Int64,
                                 r::Int64,
                                 k::Int64,
                                 l::Int64)

        # Check feasibility of sketch parameters
        if !(r<=k<=l && k<=n && l<=m)
            error("Infeasible sketch parameters r, k, and l")
        end

        # Draw and fix (complex) Gaussian normal matrices
        Omega = randn(n,k)+im*randn(n,k);
        Psi = randn(l,m)+im*randn(l,m);

        # Improve numerical stability
        Omega = qr(Omega)[1];
        Psi = qr(Psi')[1]';

        # Initialize to zero sketch
        Y = zeros(Complex{Float64},m,k);
        W = zeros(Complex{Float64},l,n);

        new(Y,W,Omega,Psi,r)

    end # function ComplexMatrixSketch

end # type ComplexMatrixSketch

# Update sketch to reflect linear update A <- theta*A + eta*H
function linear_update!(Z::MatrixSketch,
                        H::Matrix{Float64},
                        theta::Float64,
                        eta::Float64)

    Z.Y = theta*Z.Y+eta*H*Z.Omega;  # Linear update to range sketch
    Z.W = theta*Z.W+eta*Z.Psi*H;    # Linear update to co-range sketch

    return

end # function linear_update!

# Update (complex) sketch to reflect linear update A <- theta*A + eta*H
function linear_update!(Z::ComplexMatrixSketch,
                        H::Matrix{Complex{Float64}},
                        theta::Complex{Float64},
                        eta::Complex{Float64})

    Z.Y = theta*Z.Y+eta*H*Z.Omega;  # Linear update to range sketch
    Z.W = theta*Z.W+eta*Z.Psi*H;    # Linear updae to co-range sketch

    return

end # function linear_update!

# Update sketch to relfect rank one update A <- (1-eta)*A + eta*u*v'
function CGM_update!(Z::MatrixSketch,
                     u::Vector{Float64},
                     v::Vector{Float64},
                     eta::Float64)

    Z.Y = (1-eta)*Z.Y+eta*u*(v'*Z.Omega);
    Z.W = (1-eta)*Z.W+eta*(Z.Psi*u)*v';

    return

end # function CGM_update!

# Update (complex) sketch to relfect rank one update A <- (1-eta)*A + eta*u*v'
function CGM_update!(Z::ComplexMatrixSketch,
                     u::Vector{Complex{Float64}},
                     v::Vector{Complex{Float64}},
                     eta::Float64)

    Z.Y = (1-eta)*Z.Y+eta*u*(v'*Z.Omega);
    Z.W = (1-eta)*Z.W+eta*(Z.Psi*u)*v';

    return

end # function CGM_update!

# Returns factors Q (m-by-k, orthonormal columns) and X (k-by-n) such that Q*X
# forms a rank-k approximation of the sketched matrix A
function low_rank_approx(Z::MatrixSketch)

    Q = qr(Z.Y)[1];     # Orthobasis for range of Y
    U,T = qr(Z.Psi*Q);  # Orthogonal-triangular factorization
    X = T\(U'*Z.W);     # Pseudo-inverse via back substitution

    return Q,X

end # function low_rank_approx

# Returns factors Q (m-by-k, orthonormal columns) and X (k-by-n) such that Q*X
# forms a rank-k approximation of the sketched (complex) matrix A
function low_rank_approx(Z::ComplexMatrixSketch)

    Q = qr(Z.Y)[1];     # Orthobasis for range of Y
    U,T = qr(Z.Psi*Q);  # Orthogonal-triangular factorization
    X = T\(U'*Z.W);     # Pseudo-inverse via back substitution

    return Q,X

end # function low_rank_approx

# Returns factors U (n-by-2k, orthonormal columns) and S (2k conjugate
# symmetric) such that U*S*U' forms a rank-2k conjugate symmetric approximation
# of the sketched matrix A
function low_rank_symm_approx(Z::MatrixSketch)

    m = size(Z.Psi)[2];
    n,k = size(Z.Omega);

    # Check that matrix is square
    if m != n
        error("Sketched matrix is not square")
    end

    Q,X = low_rank_approx(Z);
    U,T = qr([Q X']);           # Orthogonal factorization of concatenation
    T1 = T[:,1:k];              # Extract submatrix T1
    T2 = T[:,k+1:2*k];          # Extract submatrix T2
    S = (T1*T2'+T2*T1')/2;      # Symmetrize

    return U,Symmetric(S)

end # function low_rank_symm_approx

# Returns factors U (n-by-2k, orthonormal columns) and S (2k conjugate
# symmetric) such that U*S*U' forms a rank-2k conjugate symmetric approximation
# of the sketched (complex) matrix A
function low_rank_symm_approx(Z::ComplexMatrixSketch)

    m = size(Z.Psi)[2];
    n,k = size(Z.Omega);

    # Check that matrix is square
    if m != n
        error("Sketched matrix is not square")
    end

    Q,X = low_rank_approx(Z);
    U,T = qr([Q X']);           # Orthogonal factorization of concatenation
    T1 = T[:,1:k];              # Extract submatrix T1
    T2 = T[:,k+1:2*k];          # Extract submatrix T2
    S = (T1*T2'+T2*T1')/2;      # Symmetrize

    return U,Hermitian(S)

end # function low_rank_symm_approx

# Returns factors U (n-by-2k, orthonormal columns) and d (2k, diagonal) such
# that U*diag(d)*U' forms a rank-2k positive semidefinite approximation of the
# sketched matrix A
function low_rank_psd_approx(Z::MatrixSketch)

    U,S = low_rank_symm_approx(Z);
    d,V = eig(S);                       # Form eigendecomposition
    sorted_ind = sortperm(d,rev=false); # Sort eigenvalues
    U = U*V[:,sorted_ind];              # Consolidate orthonormal factors
    d = max.(d[sorted_ind],0.0);        # Zero out negative eigenvalues

    return U,d

end # function low_rank_psd_approx

# Returns factors U (n-by-2k, orthonormal columns) and d (2k, diagonal) such
# that U*diag(d)*U' forms a rank-2k positive semidefinite approximation of the
# sketched (complex) matrix A
function low_rank_psd_approx(Z::ComplexMatrixSketch)

    U,S = low_rank_symm_approx(Z);
    d,V = eig(S);                       # Form eigendecomposition
    sorted_ind = sortperm(d,rev=false); # Sort eigenvalues
    U = U*V[:,sorted_ind];              # Consolidate orthonormal factors
    d = max.(d[sorted_ind],0.0);        # Zero out negative eigenvalues

    return U,d

end # function low_rank_psd_approx

# Returns factors Q (m-by-r, orthonormal columns), V (n-by-r, orthonormal
# columns), and S (r, diagonal) such that Q*S*V' forms a rank-r
# approximation fo the sketched matrix A
function fixed_rank_approx(Z::MatrixSketch)

    Q,X = low_rank_approx(Z);
    F = svds(X,nsv=Z.r)[1];     # Truncated SVD
    Q = Q*F[:U];                # Consolidate orthonormal factors

    return Q,F[:S],F[:V]

end # function fixed_rank_approx

# Returns factors Q (m-by-r, orthonormal columns), V (n-by-r, orthonormal
# columns), and S (r, diagonal) such that Q*S*V' forms a rank-r
# approximation fo the sketched (complex) matrix A
function fixed_rank_approx(Z::ComplexMatrixSketch)

    Q,X = low_rank_approx(Z);
    F = svds(X,nsv=Z.r)[1];     # Truncated SVD
    Q = Q*F[:U];                # Consolidate orthonormal factors

    return Q,F[:S],F[:V]

end # function fixed_rank_approx

# Returns factors U (n-by-r, orthonoral columns) and d (r, diagonal) such that
# U*diag(d)*U' forms a rank-r conjugate symmetric approximation of the sketched
# matrix A
function fixed_rank_symm_approx(Z::MatrixSketch)

    U,S = low_rank_symm_approx(Z);
    d,V = eigs(S,nev=Z.r,which=:LM);    # Truncated EVD
    U = U*V;                            # Consolidate orthonormal factors

    return U,d

end # function fixed_rank_symm_approx

# Returns factors U (n-by-r, orthonoral columns) and d (r, diagonal) such that
# U*diag(d)*U' forms a rank-r conjugate symmetric approximation of the sketched
# (complex) matrix A
function fixed_rank_symm_approx(Z::ComplexMatrixSketch)

    U,S = low_rank_symm_approx(Z);
    d,V = eigs(S,nev=Z.r,which=:LM);    # Truncated EVD
    U = U*V;                            # Consolidate orthonormal factors

    return U,d

end # function fixed_rank_symm_approx

# Returns factors U (n-by-r, orthonormal colunns) and d (r, diagonal) such
# that U*diag(d)*U' forms a rank-r positive semidefinite approximation of the 
# sketched matrix A
function fixed_rank_psd_approx(Z::MatrixSketch)

    U,S = low_rank_symm_approx(Z);
    d,V = eigs(S,nev=Z.r,which=:LR);    # Truncated EVD
    U = U*V;                            # Consolidate orthonormal factors
    d = max.(d,0.0);                    # Zero out negative eigenvalues

    return U,d

end # function fixed_rank_psd_approx

# Returns factors U (n-by-r, orthonormal colunns) and d (r, diagonal) such
# that U*diag(d)*U' forms a rank-r positive semidefinite approximation of the
# sketched (complex) matrix A
function fixed_rank_psd_approx(Z::ComplexMatrixSketch)

    U,S = low_rank_symm_approx(Z);
    d,V = eigs(S,nev=Z.r,which=:LR);    # Truncated EVD
    U = U*V;                            # Consolidate orthonormal factors
    d = max.(d,0.0);                    # Zero out negative eigenvalues

    return U,d

end # function fixed_rank_psd_approx

end # module Sketch