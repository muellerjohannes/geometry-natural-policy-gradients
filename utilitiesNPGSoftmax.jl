using LinearAlgebra
using ForwardDiff

### Tabular softmax parametrization

# Define the tabular softmax policy parametrization
function softmaxPolicy(θ)
    θ = reshape(θ, (nA, nO))
    π = exp.(θ)
    for o in 1:nO
        π[:,o] = π[:,o] / sum(π[:,o])
    end
    return π
end

# Reward function for the softmax model
function softmaxReward(θ, α, β, γ, μ, r)
    π = softmaxPolicy(θ)
    return R(π, α, β, γ, μ, r)
end

function softmaxStateActionFrequency(θ, α, β, γ, μ, r)
    π = softmaxPolicy(θ)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ)\((1-γ)*μ)
    η = Diagonal(ρ) * transpose(π)
    return η
    #return softmaxStateActionFrequency(π, α, β, γ, μ, r)
end

### Functions for policy gradient methods

# Implement variants of the natural gradient
saf(θ) = softmaxStateActionFrequency(θ, α, β, γ, μ, r)
jacobianStateActionFrequencies = θ -> ForwardDiff.jacobian(saf, θ)

# Define the conditioner of the Kakade natural gradient
logLikelihoods(θ) = log.(softmaxPolicy(θ))
jacobianLogLikelihoods = θ -> ForwardDiff.jacobian(logLikelihoods, θ)

jacobianSoftmax = θ -> ForwardDiff.jacobian(softmaxPolicy, θ)

function kakadeConditioner(θ)
    η = reshape(softmaxStateActionFrequency(θ, α, β, γ, μ, r), nS*nA)
    J = jacobianLogLikelihoods(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end

function kakadePenalty(θ)
    π = softmaxPolicy(θ)
    τ = π * β
    η = softmaxStateActionFrequency(θ, α, β, γ, μ, r)
    return -sum(log.(τ).*transpose(η))
end

# Define the conditioner of the Morimura natural gradient
logLikelihoodsSAF(θ) = log.(softmaxStateActionFrequency(θ, α, β, γ, μ, r))
jacobianLogLikelihoodsSAF = θ -> ForwardDiff.jacobian(logLikelihoodsSAF, θ)

function morimuraConditioner(θ)
    η = reshape(softmaxStateActionFrequency(θ, α, β, γ, μ, r), nS*nA)
    J = jacobianLogLikelihoodsSAF(θ)
    G = [sum(J[:, i].*J[:, j].*η) for i in 1:nP, j in 1:nP]
    return G
end

function morimuraPenalty(θ)
    η = softmaxStateActionFrequency(θ, α, β, γ, μ, r)
    return -sum(log.(η).*η)
end

# Define the σ-conditioner and corresponding penalization

function sigmaConditioner(θ, σ=1)
    η = reshape(saf(θ), nS*nA)  
    J = jacobianStateActionFrequencies(θ)
    G = transpose(J) * diagm(η.^-σ) * J
    return G
end

function sigmaPenalty(θ, σ=1)
    η = reshape(saf(θ), nS*nA)  
    if σ == 1
        return -sum(log.(η).*η)
    elseif σ == 0
        return -sum(log.(η))
    else
        return -sum(η.^-σ)
    end
end

