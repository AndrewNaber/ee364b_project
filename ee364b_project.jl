include("sketch.jl")
include("sketchyCGM.jl");
include("constrainedsketchyCGM.jl")

import Sketch
import SketchyCGM
import ConstrainedSketchyCGM

# Generates and approximately solves an instance of the partitioning problem
#
#   min. x'Wx
#   s.t. x_i = -1 or +1
#
# It forms the semidefinite relaxation
#
#   min. tr(WZ)
#   s.t. Z_ii = 1 for i = 1,...,n
#
# where Z is positive semidefinite.  This relaxation is solved using the 
# sketchy CGM method and augmented Lagrangian method.
function demo_partition()

  # Problem data
  n = 100;
  W = sprandn(n,n,0.1); W = 0.5*(W+W');

  # Set up and solve SDP relaxation using the sketching methods
  A(u,v) = [dot(u,W*v)];
  adjA(z) = z[1]*W;
  a = zeros(1);
  B(u,v) = u.*v;
  adjB(z) = diagm(z);
  b = ones(n);
  d = n*1.0;
  p_A = 1;
  p_B = n;
  f(z) = sum(z);
  gradf(z) = ones(length(z));
  r = 5;  # Target rank of the sketch
  F = ConstrainedSketchyCGM.aff_eq_symm_sketchy_CGM(f,gradf,A,adjA,a,
                                                    B,adjB,b,d,n,p_A,p_B,r);

  # Unpack sketch and compute the objective value achieved by the sketch of
  # the optimal point
  Q,lambda = Sketch.fixed_rank_psd_approx(F[1]);
  X = Q*diagm(lambda)*Q';
  obj_attained = vecdot(W,X);
  obj_actual = F[2];
  println("obj. attained by sketch: $(obj_attained)")
  println("obj. actual:             $(obj_actual)")

  return

end # function demo_partition

srand(0);
demo_partition();