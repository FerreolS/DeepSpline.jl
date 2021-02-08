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

# Aliases
export deepspline

struct deepspline
    N::Integer # Number of activation function
    K::Integer # Number of nodes
    FirstIndexes::Array{Int32} # index of the first node of each activation function
    coefs::Array{Float32}
    T::Float32 # step
    Kmin::Integer 
    Kmax::Integer 
end

# Default node between -1 and 1.
deepspline( K::Integer) = deepspline(1,K)

# Default node between -1 and 1.
deepspline(N::Integer, K::Integer) = deepspline(N,K,2f0 /(K-3))

function deepspline(N::Integer, K::Integer,T::Float32)
    @assert isodd(K) "K must be an odd number!"
    Kmin = -(K-1)/2+1;
    Kmax = (K-1)/2-1;
    
    coefs = zeros(Float32,K,N);
    for n in 1:N
        coefs[:,n] = BsplineCoef_relu(K,T)
        #= if isodd(n)
            coefs[:,n] = BsplineCoef_relu(K,T)
        else
            coefs[:,n] = BsplineCoef_abs(K,T)
        end =#
    end
    FirstIndexes = Int32.(collect(((1:N) .- 1 ).*K));
    deepspline(N, K,FirstIndexes,coefs,T,Kmin, Kmax)
end

function BsplineCoef_abs(K::Integer,T::Float32)
    @assert isodd(K) "K must be an odd number!"
    K0 = Integer((K-1)/2+1);
    coefs = zeros(Float32,K);
    coefs[K0+1:end] .= T;
    coefs[K0+1:end] = cumsum(coefs[K0+1:end]);
    coefs[1:K0-1] = coefs[end:-1:K0+1];
    return coefs
end

function BsplineCoef_relu(K::Integer,T::Float32)
    @assert isodd(K) "K must be an odd number!"
    K0 = Integer((K-1)/2+1);
    coefs = zeros(Float32,K);
    coefs[K0+1:end] .= T;
    coefs[K0+1:end] = cumsum(coefs[K0+1:end]);
    return coefs
end

function BsplineCoef_id(K::Integer,T::Float32)
    @assert isodd(K) "K must be an odd number!"
    K0 = Integer((K-1)/2+1);
    coefs = zeros(Float32,K);
    coefs[K0+1:end] .= T;
    coefs[K0+1:end] = cumsum(coefs[K0+1:end]);
    coefs[1:K0-1] = -coefs[end:-1:K0+1];
    return coefs
end

function BsplineCoef_soft(K::Integer,T::Float32)
    @assert isodd(K) "K must be an odd number!"
    K0 = Integer((K-1)/2+1);
    coefs = zeros(Float32,K);
    coefs[K0+1:end] .= T;
    coefs[K0+1:end] = max.(0f0,cumsum(coefs[K0+1:end]).-0.5f0);
    coefs[1:K0-1] = -coefs[end:-1:K0+1];
    return coefs
end
function (self::deepspline)(x::AbstractArray) 
    coefs,N, K,FirstIndexes,T,Kmin, Kmax = self.coefs, self.N, self.K, self.FirstIndexes, self.T,self.Kmin, self.Kmax    

    #k = collect(knots);
    # Linear extrapolations:
    # f(x_left) = leftmost coeff value + left_slope * (x - leftmost coeff)
    # f(x_right) = second rightmost coeff value + right_slope * (x - second rightmost coeff)
    # where the first components of the sums (leftmost/second rightmost coeff value)
    # are taken into account in DeepBspline_Func() and linearExtrapolations adds the rest.
    leftmost_slope = (coefs[2,:] .- coefs[1,:]) ./ T;
    rightmost_slope = (coefs[end,:] .- coefs[end - 1,:]) ./ T;
    leftExtrapolations  = min.((x .- Kmin.*T), 0f0) .* leftmost_slope;
    rightExtrapolations  = max.((x .- Kmax.*T), 0f0) .* rightmost_slope;
    linearExtrapolations = leftExtrapolations .+ rightExtrapolations;
    

    # First, we clamp the input to the range [leftmost coefficient, second righmost coefficient].
    # We have to clamp, on the right, to the second righmost coefficient, so that we always have
    # a coefficient to the right of x_clamped to compute its output.
    # For the values outside the range, linearExtrapolations will add what remains
    # to compute the final output of the activation, taking into account the slopes
    # on the left and right.
    xc = clamp.(x, Kmin.*T, Kmax.*T);
     # This gives the indexes (in coefficients_vect) of the left coefficients
    indexes  = (xc .- (Kmin-1).*T) ./ T .+ 1f0 ;
    floored_indexes = Int32.(floor.(indexes));
    fracs = indexes .- floored_indexes ;
    floored_indexes = floored_indexes.+FirstIndexes;
    activation_output = coefs[floored_indexes .+ 1 ] .* fracs .+ coefs[floored_indexes] .* (1f0 .- fracs) .+ linearExtrapolations; 

    return activation_output
end

function TV2(self::deepspline)
    coefs = self.coefs;
    D2 = self.coefs[1:end-2,:] .- 2f0 *self.coefs[2:end-1,:] .+ self.coefs[3:end,:] 
    return(sum(abs.(D2)));
end




Flux.@functor deepspline (coefs,)
Flux.trainable(m::deepspline) = (m.coefs,)