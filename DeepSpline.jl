""" This code implements linear splines activation functions.
A linear spline activation with parameters {a_k} and b1, b0, with knots placed
on a grid of spacing T, is described as:
deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
The ReLU representation is not well-conditioned and leads to an exponential growth
with the number of coefficients of the computational and memory requirements for
training the network.
In this module, we use an alternative B1 spline representation for the activations.
the number of b-spline coefficients exceed the number of ReLU coefficients by 2,
such that len(a) + len((b1, b_0)) = len(c), so we have the same total amount of parameters.
The coefficients of the ReLU can be computed via:
a = Lc, where L is a second finite difference matrix.
This additional number of B1 spline coefficients (2), compared to the ReLU,
allows the unique specification of the linear term term, which is in the
nullspace of the L second finite-difference matrix.
In other words, two sets of coefficients [c], [c'] which are related by
a linear term, give the same ReLU coefficients [a].
Outside a region of interest, the activation is computed via left and right
linear extrapolations using the two leftmost and rightmost coefficients, respectively.
The regularization term applied to this function is:
TV(2)(deepsline) = ||a||_1 = ||Lc||_1
For the theoretical motivation and optimality results,
please see https://arxiv.org/abs/1802.09210.
"""

using Flux
using Flux: binarycrossentropy

# Aliases
export deepspline

struct deepspline
    coefs::Array{Float32}
    # knots::LinRange
    knots::Array{Float32}
end

deepspline(in::Integer, knots) =
  deepspline(randn(Float32,in), knots)
  
function deepspline( K::Integer)
    T = 2 ./ (K-2);
    knots = collect(LinRange(-1-T, 1+T, K));
    coefs = max.(knots,0);
    coefs[2:end] = coefs[2:end] + coefs[1:end-1]
    coefs = coefs ./ coefs[end-1];
    deepspline(coefs, knots)
end
function (m::deepspline)(x::AbstractArray) 
    knots, coefs = m.knots, m.coefs;
    step = knots[2] - knots[1];

    #k = collect(knots);
    # Linear extrapolations:
    # f(x_left) = leftmost coeff value + left_slope * (x - leftmost coeff)
    # f(x_right) = second rightmost coeff value + right_slope * (x - second rightmost coeff)
    # where the first components of the sums (leftmost/second rightmost coeff value)
    # are taken into account in DeepBspline_Func() and linearExtrapolations adds the rest.
    leftmost_slope = (coefs[2] .- coefs[1]) ./ (knots[2] - knots[1]);
    rightmost_slope = (coefs[end] .- coefs[end - 1]) ./ (knots[end] - knots[end-1]);
    leftExtrapolations  = min.((x .- knots[2]), 0f0) .* leftmost_slope;
    rightExtrapolations  = max.((x .- knots[end-1]), 0f0) .* rightmost_slope;
    linearExtrapolations = leftExtrapolations .+ rightExtrapolations;
    

    # First, we clamp the input to the range [leftmost coefficient, second righmost coefficient].
    # We have to clamp, on the right, to the second righmost coefficient, so that we always have
    # a coefficient to the right of x_clamped to compute its output.
    # For the values outside the range, linearExtrapolations will add what remains
    # to compute the final output of the activation, taking into account the slopes
    # on the left and right.
    xc = clamp.(x, knots[2], knots[end-1]);
     # This gives the indexes (in coefficients_vect) of the left coefficients
   # indexes  = (xc .- knots[1]) ./ step .+ 1f0 ;
   # floored_indexes = Int64.(floor.(indexes));
    #fracs = indexes .- floored_indexes ;
    #activation_output = coefs[floored_indexes .+ 1 ] .* fracs .+ coefs[floored_indexes] .* (1f0 .- fracs) .+ linearExtrapolations; 

    floored_indexes = searchsortedlast(knots, xc);
    activation_output = coefs[floored_indexes] .* (xc .- knots[floored_indexes]) .+ coefs[floored_indexes] .* (knots[floored_indexes+1].-xc) .+ linearExtrapolations; 

    return activation_output
end

Flux.@functor deepspline (coefs,)
Flux.trainable(m::deepspline) = (m.coefs,)