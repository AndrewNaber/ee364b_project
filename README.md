# ee364b_project

The script "ee364b_project.jl" will generate and solve (using the sketch-based method) a random instance of the partitioning problem.

The implementation consists of three modules as well as a simple demonstration script.  The first module is "Sketch.jl" which defines a MatrixSketch data type and consists of sketch-related methods.  The second module is "SketchyCGM.jl" which implements the SketchCG method.  The third module is "ConstrainedSketchyCGM.jl" which implements the augmented Lagrangian method and calls the SketchCG method in "SketchyCGM.jl".
