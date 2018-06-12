module SketchyCGM

import Sketch

# Line search using the golden ratio method
function line_search(f::Function)

    tau = (3-sqrt(5))/2;

    a = 0.0;
    a_bar = 1.0;

    while abs(a_bar-a) > 1e-15
        b = a+tau*(a_bar-a);
        b_bar = a_bar-tau*(a_bar-a);
        t1 = f(b);
        t2 = f(b_bar);
        if t1 < t2
            if f(a) <= t1
                a_bar = b;
            else
                a_bar = b_bar;
            end
        elseif t1 > t2
            if t2 >= f(a_bar)
                a = b_bar;
            else
                a = b;
            end
        else
            a = b;
            a_bar = b_bar;
        end
    end

    return (a+a_bar)/2

end # function line_search

function display_update(iteration::Int64,
												objective_value::Float64,
												gap::Float64,
												elapsed_time::Float64)

	print(@sprintf "%11d" iteration)
	print(" | ")
	print(@sprintf "%14e" objective_value)
	print("   ")
	print(@sprintf "%14e" gap)
	print("   ")
	println(@sprintf "%14e" elapsed_time)

	return

end # function display_update

# Solves the problem
#
# 	min. f(GX+g)
#		s.t. tr(X) <= d
#
# where X is n-by-n and positive semidefinite.  GX+g is a p-vector.  The 
# integer r is the target rank of the sketch. The function f must be 
# differentiable and convex. If X (and z = GX+g) are not supplied, then the
# conditional gradient iterations are started from 0 (and z = g). The default
# tolerance is 1e-3, and the default MAX_ITERS is 5000. The verbose input can
# be used to toggle the output.
function symm_sketchy_CGM(f::Function,
						  						gradf::Function,
                          G::Function,
                          adjG::Function,
                          g::Vector{Float64},
						  						d::Float64,
                          n::Int64,
                          p::Int64,
						  						r::Int64;
						  						eps::Float64=1e-3,
						  						MAX_ITERS::Int64=round(Int64,5/eps),
						  						verbose::Bool=true,
						  						X::Sketch.MatrixSketch=
						  							 Sketch.MatrixSketch(n,n,r,2*r+1,4*r+3),
						  						z::Vector{Float64}=zeros(p)+g)

	# Display preamble
	if verbose
		println("================================================================")
		println("Sketchy Conditional Gradient Method (Symmetric)")
		println("----------------------------------------------------------------")
		println("MAX_ITERS = $(MAX_ITERS), eps = $(eps)")
		println("----------------------------------------------------------------")
		println(" Iterations | Objective Value |      Gap      |     Time (s)    ")
		println("----------------------------------------------------------------")
	end

	# Start timer
	ref_time = time();

	obj_val = NaN;		# Track objective value
	gap = Inf;				# Track gap
	ii = MAX_ITERS;		# Track iterations taken

	for tt = 1:MAX_ITERS

		# Calculate current objective value
		obj_val = f(z);

    # Calculate update direction
		F = eigs(adjG(gradf(z)),nev=1,which=:SR,
						 maxiter=1000000,tol=1e-10);
		lambda = F[1][1]; u = F[2][:];
    if lambda <= 0
      h = G(d*u,u)+g;
    else
      h = zeros(p)+g;
    end

    # Calculate gap
    gap = min(dot(z-h,gradf(z)),gap);

    # Display progress
		if verbose && rem(tt-1,100) == 0
			display_update(tt-1,obj_val,gap,time()-ref_time)
		end

		# Check stopping criterion
		if gap <= eps
			ii = tt-1;
			break
		end

		# Choose step size
		f_ls(eta) = f((1-eta)*z+eta*h);
		eta = line_search(f_ls);

		# Update iterate and sketches
		z = (1-eta)*z+eta*h;
    if lambda <= 0
      Sketch.CGM_update!(X,d*u,u,eta);
    else
      Sketch.CGM_update!(X,zeros(n),zeros(n),eta);
    end

	end

	if verbose
		println("----------------------------------------------------------------")
		if ii == MAX_ITERS
			println("Status: Failure to converge. Solution may be inaccurate.")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total time (s):        ",@sprintf "%14e" time()-ref_time)
		else
			println("Status: Solved")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		end
		println("================================================================")
	end

	return X,obj_val,gap,z

end # function symm_sketchy_CGM

# Solves the problem
#
# 	min. f(GX+g)
#		s.t. ||X||_2* <= d
#
# where X is m-by-n.  GX+g is a p-vector. The integer r is the target rank of
# the sketch. The function f must be differentiable and convex.  If X (and z = 
# GX+g) are not supplied, then the conditional gradient iterations are started 
# from 0 (and z = g). The default tolerance is 1e-3, and the default MAX_ITERS
# is 5000. The verbose input can be used to toggle the output.
function nonsymm_sketchy_CGM(f::Function,
							 							 gradf::Function,
							 							 G::Function,
                             adjG::Function,
                             g::Vector{Float64},
							 							 d::Float64,
                             m::Int64,
                             n::Int64,
                             p::Int64,
							 							 r::Int64;
							 							 eps::Float64=1e-3,
							 							 MAX_ITERS::Int64=round(Int64,5/eps),
							 							 verbose::Bool=true,
							 							 X::Sketch.MatrixSketch=
							 							 		 Sketch.MatrixSketch(m,n,r,2*r+1,4*r+3),
							 							 z::Vector{Float64}=zeros(p)+g)

	# Display preamble
	if verbose
		println("================================================================")
		println("Sketchy Conditional Gradient Method (Nonsymmetric)")
		println("----------------------------------------------------------------")
		println("MAX_ITERS = $(MAX_ITERS), eps = $(eps)")
		println("----------------------------------------------------------------")
		println(" Iterations | Objective Value |      Gap      |     Time (s)    ")
		println("----------------------------------------------------------------")
	end

	# Start timer
	ref_time = time();

	obj_val = NaN;		# Track objective value
	gap = Inf;				# Track gap
	ii = MAX_ITERS;		# Track iterations taken

	for tt = 1:MAX_ITERS

		# Calculate current objective value
		obj_val = f(z);

    # Calculate update direction
		F = svds(adjG(gradf(z)),nsv=1,maxiter=1000000)[1];
		u = F[:U][:]; v = F[:V][:];
		h = G(-d*u,v)+g;

		# Calculate gap
		gap = min(dot(z-h,gradf(z)),gap);

		# Display progress
		if verbose && rem(tt-1,100) == 0
			display_update(tt-1,obj_val,gap,time()-ref_time)
		end

		# Check stopping criterion
		if gap <= eps
			ii = tt-1;
			break
		end

		# Choose step size
		f_ls(eta) = f((1-eta)*z+eta*h);
		eta = line_search(f_ls);

		# Update iterate and sketches
		z = (1-eta)*z+eta*h;
		Sketch.CGM_update!(X,-d*u,v,eta);

	end

	if verbose
		println("----------------------------------------------------------------")
		if ii == MAX_ITERS
			println("Status: Failure to converge. Solution may be inaccurate.")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total time (s):        ",@sprintf "%14e" time()-ref_time)
		else
			println("Status: Solved")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		end
		println("================================================================")
	end

	return X,obj_val,gap,z

end # function nonsymm_sketchy_CGM

end # module SketchyCGM