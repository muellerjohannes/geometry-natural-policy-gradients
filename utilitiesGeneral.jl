using LinearAlgebra

# Reward via the expression as determinantal rational function
function R_det(π, α, β, γ, μ, r)
    #π = reshape(π, (nA, nO))
    τ = π * β  # Observation to state policy
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * τ)  # Compute the one step reward
    counter = det(I - γ * pπ + μ * transpose(rπ))
    denominator = det(I - γ * pπ)
    return (1-γ) * (counter/denominator - 1)
end

# Reward of a policy
function R(π, α, β, γ, μ, r)
    τ = π * β  # Observation to state policy
    # Compute the state-state transition
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * τ)  # Compute the one step reward
    Vπ = (1-γ)*(I-γ*transpose(pπ))\rπ  # Compute the state value function via Bellman's equation
    return sum(μ.*Vπ) 
end

# Value of a policy
function valueFunction(π, α, β, γ, r)
    (nS, nA) = size(r)
    nO = size(β)[1]
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    rπ = diag(r * π * β)
    Vπ = pinv(I - γ * transpose(pπ)) * (1-γ) * rπ
    return Vπ
end

# State action frequency for a policy
 function stateActionFrequency(π, α, β, γ, μ, r)
    τ = π * β
    pπ = [transpose(τ[:,s_old]) * α[s_new,s_old,:] for s_new in 1:nS, s_old in 1:nS]
    ρ = (I-γ*pπ)\((1-γ)*μ)
    η = Diagonal(ρ) * transpose(τ)
    return η
end

### Geometry of state-action frequencies

# Linear equalities
function linearEqualities(η, μ, γ, α)
    nS = size(α)[1]
    linEq = [sum(η[s, :]) - γ*(dot(η[:, :], α[s, :, :])) - (1-γ)*μ[s] for s in 1:nS]
    if γ == 1
        linEq = append!(linEq, sum(η) - 1)
    end
    return linEq
end

# Compute the observation policy from the state action distribution
function observationPolicy(η, β)
    (nO, nS) = size(β)
    if rank(β) < nO
        error("observation mechanism does not satisfy the rank condition")
        return
    end
    # Compute the state policy
    τ = zeros((nS, nA));
    for i = 1:nS
        τ[i, :] = η[i, :] / sum(η[i, :]);
    end
    # Compute the observation policy
    π = transpose(τ) * pinv(β)
    return π
end
