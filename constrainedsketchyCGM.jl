module ConstrainedSketchyCGM

import Sketch
import SketchyCGM

function display_update(iteration::Int64,
												objective_value::Float64,
												res::Float64,
												elapsed_time::Float64)

	print(@sprintf "%11d" iteration)
	print(" | ")
	print(@sprintf "%14e" objective_value)
	print("   ")
	print(@sprintf "%14e" res)
	print("   ")
	println(@sprintf "%14e" elapsed_time)

	return

end # function display_update

# Solves problem
#
#		min. f(AX+a)
#		s.t. tr(X) <= d
# 			 BX = b
#
# where X is n-by-n and positive semidefinite.
function aff_eq_symm_sketchy_CGM(f::Function,
																 gradf::Function,
																 A::Function,
																 adjA::Function,
																 a::Vector{Float64},
																 B::Function,
																 adjB::Function,
																 b::Vector{Float64},
																 d::Float64,
																 n::Int64,
																 p_A::Int64,
																 p_B::Int64,
																 r::Int64;
																 rho::Float64=1.1,
																 eps::Float64=1e-3,
																 MAX_ITERS::Int64=round(Int,5/eps),
																 verbose::Bool=true)

	# Display preamble
	if verbose
		println("================================================================")
		println("Augmented Lagrangian (Affine Equality Constraints)")
		println("Sketchy Conditional Gradient Method (Symmetric)")
		println("----------------------------------------------------------------")
		println("MAX_ITERS = $(MAX_ITERS), eps = $(eps), rho = $(rho)")
		println("----------------------------------------------------------------")
		println(" Iterations | Objective Value |      Res      |     Time (s)    ")
		println("----------------------------------------------------------------")
	end

	# Start timer
	ref_time = time();

	# Initialize primal variable
	X = Sketch.MatrixSketch(n,n,r,2*r+1,4*r+3);
	z = zeros(p_A+p_B)+[a; -b];

	# Initialize dual variable for equality constraints
	nu = ones(p_B);

	obj_val = NaN;		# Track objective value
	res = NaN;				# Track residual
	gap = Inf;				# Track gap
	ii = MAX_ITERS;		# Track iterations taken

	for tt = 1:MAX_ITERS

		# Set up augmented Lagrangian
		f_aug_Lag(z) = begin
		    z_A = z[1:p_A];
		    z_B = z[p_A+1:end];
		    return f(z_A)+dot(nu,z_B)+0.5*rho*norm(z_B)^2
		end # f_aug_Lag
		gradf_aug_Lag(z) = begin
		    z_A = z[1:p_A];
		    z_B = z[p_A+1:end];
		    return [gradf(z_A); nu+rho*z_B]
		end # gradf_aug_Lag
		G_aug_Lag(u,v) = [A(u,v); B(u,v)];
		adjG_aug_Lag(z) = begin
		    z_A = z[1:p_A];
		    z_B = z[p_A+1:end];
		    return  adjA(z_A)+adjB(z_B)
		end # adjG_aug_Lag
		g_aug_Lag = [a; -b];

		# Solve augmented Lagrangian and extract results
		F = SketchyCGM.symm_sketchy_CGM(f_aug_Lag,gradf_aug_Lag,
																		G_aug_Lag,adjG_aug_Lag,g_aug_Lag,d,
																		n,p_A+p_B,r,verbose=false,eps=0.1/tt,
																		X=X,z=z);
		X = F[1];
		z = F[4]; z_A = z[1:p_A]; z_B = z[p_A+1:end];
		gap = F[3];

		# Calculate current objective value
		obj_val = f(z_A);

		# Calculate residual
		res = norm(z_B);

		# Display progress
		if verbose
			display_update(tt,obj_val,res,time()-ref_time)
		end

		# Check stopping criterion
		if res <= eps
			ii = tt-1;
			break
		end

		# Update dual variables
		nu += rho*z_B;

		# Increase penalty parameter
		rho *= 1.1;

	end

	if verbose
		println("----------------------------------------------------------------")
		if ii == MAX_ITERS
			println("Status: Failed to converge. Solution may be inaccurate.")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Residual:        ",@sprintf "%14e" res)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		else
			println("Status: Solved")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Residual:        ",@sprintf "%14e" res)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		end
		println("================================================================")
	end

	return X,obj_val,res

end # function aff_eq_symm_sketchy_CGM

# Solves problem
#
#		min. f(AX+a)
#		s.t. tr(X) <= d
# 			 CX <= c
#
# where X is n-by-n and positive semidefinite.
function aff_ineq_symm_sketchy_CGM(f::Function,
																	 gradf::Function,
																	 A::Function,
																	 adjA::Function,
																	 a::Vector{Float64},
																	 C::Function,
																	 adjC::Function,
																	 c::Vector{Float64},
																	 d::Float64,
																	 n::Int64,
																	 p_A::Int64,
																	 p_C::Int64,
																	 r::Int64;
																	 rho::Float64=1.1,
																	 eps::Float64=1e-3,
																	 MAX_ITERS::Int64=round(Int,5/eps),
																	 verbose::Bool=true)

	# Display preamble
	if verbose
		println("================================================================")
		println("Augmented Lagrangian (Affine Inequality Constraints)")
		println("Sketchy Conditional Gradient Method (Symmetric)")
		println("----------------------------------------------------------------")
		println("MAX_ITERS = $(MAX_ITERS), eps = $(eps), rho = $(rho)")
		println("----------------------------------------------------------------")
		println(" Iterations | Objective Value |      Res      |     Time (s)    ")
		println("----------------------------------------------------------------")
	end

	# Start timer
	ref_time = time();

	# Initialize primal variable
	X = Sketch.MatrixSketch(n,n,r,2*r+1,4*r+3);
	z = zeros(p_A+p_C)+[a; -c];

	# Initialize dual variable for equality constraints
	lambda = ones(p_C);

	obj_val = NaN;		# Track objective value
	res = NaN;				# Track residual
	gap = Inf;				# Track gap
	ii = MAX_ITERS;		# Track iterations taken

	for tt = 1:MAX_ITERS

		# Set up augmented Lagrangian
		f_aug_Lag(z) = begin
	    z_A = z[1:p_A];
	    z_C = z[p_A+1:end];
	    return f(z_A)+(norm(max.(lambda+rho*z_C,0.0))^2-norm(lambda)^2)/(2*rho)
		end # f_aug_Lag
		gradf_aug_Lag(z) = begin
	    z_A = z[1:p_A];
	    z_C = z[p_A+1:end];
	    return [gradf(z_A); max.(lambda+rho*z_C,0.0)]
		end # gradf_aug_Lag
		G_aug_Lag(u,v) = [A(u,v); C(u,v)];
		adjG_aug_Lag(z) = begin
	    z_A = z[1:p_A];
	    z_C = z[p_A+1:end];
	    return  adjA(z_A)+adjC(z_C)
		end # adjG_aug_Lag
		g_aug_Lag = [a; -c];

		# Solve augmented Lagrangian
		F = SketchyCGM.symm_sketchy_CGM(f_aug_Lag,gradf_aug_Lag,
																		G_aug_Lag,adjG_aug_Lag,g_aug_Lag,d,
																		n,p_A+p_C,r,verbose=false,eps=0.1/tt,
																		X=X,z=z);
		X = F[1];
		z = F[4]; z_A = z[1:p_A]; z_C = z[p_A+1:end];
		gap = F[3];

		# Calculate current objective value
		obj_val = f(z_A);

		# Calculate residual
		res = norm(max.(z_C,0.0));

		# Display progress
		if verbose
			display_update(tt,obj_val,res,time()-ref_time)
		end

		# Check stopping criterion
		if res <= eps
			ii = tt-1;
			break
		end

		# Update dual variables
		lambda = max.(lambda+rho*z_C,0.0);

		# Increase penalty parameter
		rho *= 1.1;

	end

	if verbose
		println("----------------------------------------------------------------")
		if ii == MAX_ITERS
			println("Status: Failed to converge. Solution may be inaccurate.")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Residual:        ",@sprintf "%14e" res)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		else
			println("Status: Solved")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Residual:        ",@sprintf "%14e" res)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		end
		println("================================================================")
	end

	return X,obj_val,res

end # function aff_ineq_symm_sketchy_CGM

# Solves problem
#
#		min. f(AX+a)
#		s.t. tr(X) <= d
#				 BX = b
# 			 CX <= c
#
# where X is n-by-n and positive semidefinite.
function aff_const_symm_sketchy_CGM(f::Function,
																	  gradf::Function,
																	  A::Function,
																	  adjA::Function,
																	  a::Vector{Float64},
																	  B::Function,
																	  adjB::Function,
																	  b::Vector{Float64},
																	  C::Function,
																	  adjC::Function,
																	  c::Vector{Float64},
																	  d::Float64,
																	  n::Int64,
																	  p_A::Int64,
																	  p_B::Int64,
																	  p_C::Int64,
																	  r::Int64;
																	  rho::Float64=1.1,
																	  eps::Float64=1e-3,
																	  MAX_ITERS::Int64=round(Int,5/eps),
																	  verbose::Bool=true)

	# Display preamble
	if verbose
		println("================================================================")
		println("Augmented Lagrangian (Affine Equality & Inequality Constraints)")
		println("Sketchy Conditional Gradient Method (Symmetric)")
		println("----------------------------------------------------------------")
		println("MAX_ITERS = $(MAX_ITERS), eps = $(eps), rho = $(rho)")
		println("----------------------------------------------------------------")
		println(" Iterations | Objective Value |      Res      |     Time (s)    ")
		println("----------------------------------------------------------------")
	end

	# Start timer
	ref_time = time();

	# Initialize primal variable
	X = Sketch.MatrixSketch(n,n,r,2*r+1,4*r+3);
	z = zeros(p_A+p_B+p_C)+[a; -b; -c];

	# Initialize dual variable for equality constraints
	nu = ones(p_B);
	lambda = ones(p_C);

	obj_val = NaN;		# Track objective value
	res = NaN;				# Track residual
	gap = Inf;				# Track gap
	ii = MAX_ITERS;		# Track iterations taken

	for tt = 1:MAX_ITERS

		# Set up augmented Lagrangian
		f_aug_Lag(z) = begin
	    z_A = z[1:p_A];
	    z_B = z[p_A+1:p_A+p_B];
	    z_C = z[p_A+p_B+1:end];
	    return f(z_A)+dot(nu,z_B)+0.5*rho*norm(z_B)^2+
	    				(norm(max.(lambda+rho*z_C,0.0))^2-norm(lambda)^2)/(2*rho)
		end # f_aug_Lag
		gradf_aug_Lag(z) = begin
	    z_A = z[1:p_A];
	    z_B = z[p_A+1:p_A+p_B];
	    z_C = z[p_A+p_B+1:end];
	    return [gradf(z_A); nu+rho*z_B; max.(lambda+rho*z_C,0.0)]
		end # gradf_aug_Lag
		G_aug_Lag(u,v) = [A(u,v); B(u,v); C(u,v)];
		adjG_aug_Lag(z) = begin
	    z_A = z[1:p_A];
	    z_B = z[p_A+1:p_A+p_B];
	    z_C = z[p_A+p_B+1:end];
	    return  adjA(z_A)+adjB(z_B)+adjC(z_C)
		end # adjG_aug_Lag
		g_aug_Lag = [a; -b; -c];

		# Solve augmented Lagrangian
		F = SketchyCGM.symm_sketchy_CGM(f_aug_Lag,gradf_aug_Lag,
																		G_aug_Lag,adjG_aug_Lag,g_aug_Lag,d,
																		n,p_A+p_B+p_C,r,verbose=false,
																		eps=0.1/tt,X=X,z=z);
		X = F[1];
		z = F[4]; z_A = z[1:p_A]; z_B = z[p_A+1:p_A+p_B]; z_C = z[p_A+p_B+1:end];
		gap = F[3];

		# Calculate current objective value
		obj_val = f(z_A);

		# Calculate residual
		res = norm(z_B)^2+norm(max.(z_C,0.0));

		# Display progress
		if verbose
			display_update(tt,obj_val,res,time()-ref_time)
		end

		# Check stopping criterion
		if res <= eps
			ii = tt-1;
			break
		end

		# Update dual variables
		nu += rho*z_B;
		lambda = max.(lambda+rho*z_C,0.0);

		# Increase penalty parameter
		rho *= 1.1;

	end

	if verbose
		println("----------------------------------------------------------------")
		if ii == MAX_ITERS
			println("Status: Failed to converge. Solution may be inaccurate.")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Residual:        ",@sprintf "%14e" res)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		else
			println("Status: Solved")
			println("Final Objective Value: ",@sprintf "%14e" obj_val)
			println("Final Residual:        ",@sprintf "%14e" res)
			println("Final Gap:             ",@sprintf "%14e" gap)
			println("Total Time (s):        ",@sprintf "%14e" time()-ref_time)
		end
		println("================================================================")
	end

	return X,obj_val,res

end # function aff_const_symm_sketchy_CGM

end # module ConstrainedSketchyCGM