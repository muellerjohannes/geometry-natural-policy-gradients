using LinearAlgebra
using ForwardDiff
using Plots
import Plots.heatmap
using Plots.PlotMeasures
using PlotlySave
using LsqFit
include("utilitiesGeneral.jl")
include("utilitiesNPGSoftmax.jl")

### Define the MDP
begin
    # Cardinality of the state, action, observation and parameter space
    nS = 2;
    nA = 2;
    nO = 2;
    nP = nO*nA;

    # Define a random transition kernel and instantaneous reward function
    α = zeros((nS, nS, nA));
    α[:,1,:] = Matrix(I, 2, 2);
    α[:,2,:] = [0 1; 1 0];
    β = [1 0; 0 1];
    γ = 0.9;
    μ = [0.8, 0.2]
    r = [1. 0.; 2. 0.];

    #Define the parameter policy gradient
    reward(θ) = R(softmaxPolicy(θ), α, β, γ, μ, r)
    ∇R = θ -> ForwardDiff.gradient(reward, θ)

    # Define a random transition kernel and instantaneous reward function
     
    # Compute the optimal reward
    rewards_det = zeros(2,2)
    for i in 1:2
        for j in 1:2
        π = transpose([i-1 2-i; j-1 2-j])
        rewards_det[i,j] = R(π, α, β, γ, μ, r)
        end
    end
    R_opt = maximum(rewards_det)
    R_min = minimum(rewards_det)
    #Define the corners of the probability simplex required for the plotting
    Bas = [1. 1 1; -1 -1 1; -1 1 -1; 1 -1 -1];
end;

### Plot the state-action polytope and the heatmap over the policy polytope
begin
    ### Plot everything in state-action space
    # Compute the vertices of the state-action polytope
    ηDet = zeros(2, 2, 3)
    for i in 1:2
        for j in 1:2
        π = transpose([i-1 2-i; j-1 2-j])
        η = stateActionFrequency(π, α, β, γ, μ, r)
        ηDet[i, j, :] = transpose(Bas) * vec(η) #
        end
    end
    ηDet = reshape(ηDet, (4, 3))
    # Begin the plot
    p_state_action_polytope = Plots.plot(ηDet[:,1], ηDet[:,2], ηDet[:,3], seriestype=:scatter, markersize = 3, 
    color="black", # camera = (30, 0), showaxis=false, ticks=false, legend=false, size = (400, 400),
    #title="Vanilla PG", titlefontsize=20, fontfamily="Computer Modern", size = (400, 400), xlims=(-1., 1.), label=false,
    #
    #ylims=(-1.,1.), zlims=(-1.,1.), margin = -2cm
    )
    Plots.plot!(Bas[[1, 2, 3, 4, 1, 3],1], Bas[[1, 2, 3, 4, 1, 3],2], Bas[[1, 2, 3, 4, 1, 3],3], color="black", label=false, width=1.2, linestyle=:dash)
    k=2*10^2
    ηAll = zeros(k + 1, k+1, 3)
    for i in 1:(k + 1)
        for j in 1:(k + 1)
        π = transpose([(i-1)/k (k-i+1) / k; (j-1)/k (k-j+1)/k])
        η = stateActionFrequency(π, α, β, γ, μ, r)
        ηAll[i, j, :] = transpose(Bas) * vec(η) #
        end
    end
    ηAll = reshape(ηAll, ((k+1)^2, 3))
    Plots.plot!(ηAll[:,1], ηAll[:,2], ηAll[:,3],linewidth = 2, label=false, color="black", alpha=0.6)


    # Calculate things for the heatmap
    n_plot = 200
    x = range(0, 1, length = n_plot)
    y = range(0, 1, length = n_plot)
    z = zeros(n_plot, n_plot)
    for i in 1:n_plot
        for j in 1:n_plot
            π = [x[i] y[j]; 1-x[i] 1-y[j]]
            z[i, j] = R(π, α, β, γ, μ, r) 
        end
    end 

    p_heatmap = heatmap(x,y,transpose(z));

end;

### Number of training trajectories
nTrajectories = 30;
θ₀ = randn(nTrajectories, nA*nS);
nIterations = 3*10^3;

### Kakade NPG
# Define number of iterations of the gradient ascent as well as the step size
Δt = 10^-1;
@elapsed begin
    # Allocate the space for the training trajectories
    time_Kakade = zeros(nTrajectories, nIterations);
    rewardTrajectories_Kakade = zeros(nTrajectories, nIterations);
    policyTrajectories_Kakade = zeros(nTrajectories, nIterations, nA);
    ηTrajectories_Kakade = zeros(nIterations, nTrajectories, 3);
    #Optimize using Kakade natural gradient trajectories
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            policyTrajectories_Kakade[i, k,:] = π[1, :]
            rewardTrajectories_Kakade[i, k] = R(π, α, β, γ, μ, r)
            η = stateActionFrequency(π, α, β, γ, μ, r)
            ηTrajectories_Kakade[k, i, :] = transpose(Bas) * vec(η)
            Δθ = pinv(kakadeConditioner(θ)) * ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ
            if k < nIterations
                time_Kakade[i, k+1] =  time_Kakade[i, k] + stepsize
            end
        end
    end

    #=
    ### Make raw plots 
    # Make state-action plot
    p_Kakade_state_action = Plots.plot(p_state_action_polytope, ηTrajectories_Kakade[:,:,1], ηTrajectories_Kakade[:,:,2], ηTrajectories_Kakade[:,:,3], width=1.5)
    Plots.plot!(p_Kakade_state_action, ηDet[[1, 2, 4, 3, 1],1], ηDet[[1, 2, 4, 3, 1],2], ηDet[[1, 2, 4, 3, 1],3], color="black", width=1.2)
    Plots.plot!(p_Kakade_state_action, Bas[[2, 4],1], Bas[[2, 4],2], Bas[[2, 4],3], color="black", width=1.2, linestyle=:dash)

    # Plot reward trajectories
    gap = R_opt*ones(size(transpose(rewardTrajectories_Kakade[:,:,1])))-transpose(rewardTrajectories_Kakade[:,:,1])
    p_Kakade_reward = plot(transpose(time_Kakade), gap)

    # Plot the heatmap and trajectories in policy space
    p_Kakade_policies = plot(p_heatmap, transpose(policyTrajectories_Kakade[:,:,1]), transpose(policyTrajectories_Kakade[:,:,2]), width=2)=#
end

### σ-NPG
# Define range of σ and allocate storage for the trajectories
@elapsed begin
    # Define σ-s and the time speed
    sigmas = [-0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
    speeds = [10^-1, 10^-1, 10^-1, 10^-1, 2*10^-2, 10^-2, 7*10^-3, 5*10^-3]
    # Allocate space for the trajectories etc
    rewardTrajectories_σ = zeros(length(sigmas), nTrajectories, nIterations);
    policyTrajectories_σ = zeros(length(sigmas), nTrajectories, nIterations, nA);
    ηTrajectories_σ = zeros(length(sigmas), nIterations, nTrajectories, 3);
    time_σ = zeros(length(sigmas), nTrajectories, nIterations);
    # Arrays to store the raw plots in
    #p_σ_state_action = []
    #p_σ_reward = []
    #p_σ_policies = []
end;

# Run the optimization and make raw plots
@elapsed for σ in sigmas
    #Optimize using σ-NPG 
    index = findfirst(i -> i == σ, sigmas)
    Δt = speeds[index]
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            policyTrajectories_σ[index, i, k,:] = π[1, :]
            rewardTrajectories_σ[index, i, k] = R(π, α, β, γ, μ, r)
            η = stateActionFrequency(π, α, β, γ, μ, r)
            ηTrajectories_σ[index, k, i, :] = transpose(Bas) * vec(η)
            Δθ = pinv(sigmaConditioner(θ, σ)) * ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ
            if k < nIterations
                time_σ[index, i, k+1] =  time_σ[index, i, k] + stepsize
            end
        end
    end

    #=
    ### Make raw plots
    # Make state-action plot
    p = Plots.plot(p_state_action_polytope, ηTrajectories_σ[index,:,:,1], ηTrajectories_σ[index,:,:,2], ηTrajectories_σ[index,:,:,3],
     width=1.5)
    Plots.plot!(p, ηDet[[1, 2, 4, 3, 1],1], ηDet[[1, 2, 4, 3, 1],2], ηDet[[1, 2, 4, 3, 1],3], color="black", width=1.2)
    Plots.plot!(p, Bas[[2, 4],1], Bas[[2, 4],2], Bas[[2, 4],3], color="black", width=1.2, linestyle=:dash)

    push!(p_σ_state_action, p)

    # Plot reward trajectories
    gap = R_opt*ones(size(transpose(rewardTrajectories_σ[index,:,:,1])))-transpose(rewardTrajectories_σ[index,:,:,1])
    p = plot(transpose(time_σ[index,:,:]), gap)
    push!(p_σ_reward, p)

    # Plot the heatmap and trajectories in policy space
    p = plot(p_heatmap, transpose(policyTrajectories_σ[index,:,:,1]), transpose(policyTrajectories_σ[index,:,:,2]), width=2)
    push!(p_σ_policies, p)=#

end

### Vanilla PG
# Define number of iterations of the gradient ascent as well as the step size
Δt = 10^-2;
@elapsed begin
    # Allocate the space for the training trajectories
    rewardTrajectories_vanilla = zeros(nTrajectories, nIterations);
    policyTrajectories_vanilla = zeros(nTrajectories, nIterations, nA);
    ηTrajectories_vanilla = zeros(nIterations, nTrajectories, 3);
    time_vanilla = zeros(nTrajectories, nIterations);
    #Optimize using vanilla PG
    for i in 1:nTrajectories
        θ = θ₀[i,:]
        for k in 1:nIterations
            π = softmaxPolicy(θ)
            policyTrajectories_vanilla[i, k,:] = π[1, :]
            rewardTrajectories_vanilla[i, k] = R(π, α, β, γ, μ, r)
            η = stateActionFrequency(π, α, β, γ, μ, r)
            ηTrajectories_vanilla[k, i, :] = transpose(Bas) * vec(η)
            Δθ = ∇R(θ)
            stepsize = Δt / norm(Δθ)
            θ += stepsize * Δθ
            if k < nIterations
                time_vanilla[i, k+1] =  time_vanilla[i, k] + stepsize
            end
        end
    end

    #=
    ### Make raw plots 
    # Make state-action plot
    p_vanilla_state_action = Plots.plot(p_state_action_polytope, ηTrajectories_vanilla[:,:,1], ηTrajectories_vanilla[:,:,2], ηTrajectories_vanilla[:,:,3], width=1.5)
    Plots.plot!(p_vanilla_state_action, ηDet[[1, 2, 4, 3, 1],1], ηDet[[1, 2, 4, 3, 1],2], ηDet[[1, 2, 4, 3, 1],3], color="black", width=1.2)
    Plots.plot!(p_vanilla_state_action, Bas[[2, 4],1], Bas[[2, 4],2], Bas[[2, 4],3], color="black", width=1.2, linestyle=:dash)

    # Plot reward trajectories
    gap = R_opt*ones(size(transpose(rewardTrajectories_vanilla[:,:,1])))-transpose(rewardTrajectories_vanilla[:,:,1])
    p_vanilla_reward = plot(transpose(time_vanilla), gap)

    # Plot the heatmap and trajectories in policy space
    p_vanilla_policies = plot(p_heatmap, transpose(policyTrajectories_vanilla[:,:,1]), transpose(policyTrajectories_vanilla[:,:,2]), width=2)
    =#
end

########### Plotting #############

# Collect the plotting data 
begin
    
    times_all = zeros(length(sigmas)+2, nTrajectories, nIterations)
    times_all[1,:,:] = time_vanilla
    times_all[2,:,:] = time_Kakade
    times_all[3:end,:,:] = time_σ
    
    rewards_all = zeros(length(sigmas)+2, nTrajectories, nIterations)
    rewards_all[1,:,:] = rewardTrajectories_vanilla
    rewards_all[2,:,:] = rewardTrajectories_Kakade
    rewards_all[3:end,:,:] = rewardTrajectories_σ    

    policies_all = zeros(length(sigmas)+2, nTrajectories, nIterations, nA)
    policies_all[1,:,:,:] = policyTrajectories_vanilla
    policies_all[2,:,:,:] = policyTrajectories_Kakade
    policies_all[3:end,:,:,:] = policyTrajectories_σ   

    ηs_all = zeros(length(sigmas)+2, nIterations, nTrajectories, 3)
    ηs_all[1,:,:,:] = ηTrajectories_vanilla
    ηs_all[2,:,:,:] = ηTrajectories_Kakade
    ηs_all[3:end,:,:,:] = ηTrajectories_σ   
    
end;

# Titles for the plots and set the font sizes
titles = ["Vanilla PG", "Kakade's NPG", "\$σ=-0.5\$", "\$σ=0\$ (Euclidean)", "\$σ=0.5\$", "\$σ=1\$ (Fisher/Morimura)", "\$σ=1.5\$", 
"\$σ=2\$ (Itakura-Saito)", "\$σ=3\$", "\$σ=4\$"]
title_fontsize, tick_fontsize, legend_fontsize, guide_fontsize = 18, 14, 14, 14;

### Plot state-action trajectories
plots_state_action = []
for i in 1:(length(sigmas)+2)
    # State-action plot
    p = plot(p_state_action_polytope, ηs_all[i,:,:,1], ηs_all[i,:,:,2], ηs_all[i,:,:,3], width=1.5,
    title = titles[i], fontfamily="Computer Modern", camera = (30, 0), showaxis=false, ticks=false, legend=false, 
    titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, 
    guidefontsize=guide_fontsize, size = (400, 400), ylims=(-1.,1.), zlims=(-1.,1.), margin = -2cm)
    Plots.plot!(p, ηDet[[1, 2, 4, 3, 1],1], ηDet[[1, 2, 4, 3, 1],2], ηDet[[1, 2, 4, 3, 1],3], color="black", width=1.2)
    Plots.plot!(p, Bas[[2, 4],1], Bas[[2, 4],2], Bas[[2, 4],3], color="black", width=1.2, linestyle=:dash)

    push!(plots_state_action, p)
    
    save("Mathematik/POMDPs/Coding/Julia/POMDPs/natural-policy-gradients/graphics/state-action-$i.pdf", plots_state_action[i])
    
end

### Make plots of the policy trajectories
plots_policies = []
for i in 1:(length(sigmas)+2)

    # Policy plot
    p = plot(p_heatmap, transpose(policies_all[i,:,:,1]), transpose(policies_all[i,:,:,2]), width=2, legend=false, 
    linewidth=1.5, aspect_ratio=:equal, titlefontsize=title_fontsize, tickfontsize=tick_fontsize, 
    legendfontsize=legend_fontsize, guidefontsize=guide_fontsize, fontfamily="Computer Modern", size = (400, 400),
     framestyle=:box, xticks = 0:1:1, yticks = 0:1:1, xlims=(0,1), ylims=(0,1), title = titles[i], colorbar=false)

    push!(plots_policies, p)

    # Insert path and comment out for exporting figure as a pdf
    # save("INSERT_PATH/graphics/policies-$i.pdf", plots_policies[i])
end

### Make plots of the rewards
# Define axis scaling (unscaled, log or log-log) and predicted sublinear convergence rates
begin
    exponent = vcat([1, 0], [1/(σ-1) for σ in sigmas])
    x_axis_scaling = [:log, :linear, :linear, :linear, :linear, :linear, :log, :log, :log, :log]
    y_axis_scaling = [:log, :log, :linear, :linear, :linear, :log, :log, :log, :log, :log]    
end

# Do all plots
plots_reward = []
for i in 1:(length(sigmas)+2)
    times_all[i,:,1] = minimum(times_all[i,:,2:end])*ones(nTrajectories)    
    gap = R_opt*ones(size(transpose(rewards_all[i,:,:,1])))-transpose(rewards_all[i,:,:,1]);
    # State-action plot
    p = plot(transpose(times_all[i,:,:]), gap, linewidth=1.5)  
    if x_axis_scaling[i] == :log
        t = range(minimum(times_all[i,:,1:10^3]), maximum(times_all[i,:,1:10^3]), 10)
        # Model fit for the scaling parameter of the predicted sublinear decay O(t^-τ)
        horizon=1001:3000
        model(t, p) = p[1] * t.^-exponent[i]
        p0 = [1.]
        fit = curve_fit(model, times_all[i,1,horizon], gap[horizon, 1], p0)
        param = fit.param[1]
        plot!(p, t, param*t.^-exponent[i], linewidth = 4, color="black", alpha = 0.5, linestyle=:dash)
    end
    plot!(p, legend = false, title=titles[i], linewidth=1., size=(400,300), fontfamily="Computer Modern", 
    titlefontsize=title_fontsize, tickfontsize=tick_fontsize, legendfontsize=legend_fontsize, guidefontsize=guide_fontsize,
    framestyle=:box, xaxis=x_axis_scaling[i], yaxis=y_axis_scaling[i],
    ylims=(minimum(gap[10^3,:]),1.2*maximum(gap)), xlims=(minimum(times_all[i,:,1:10^3]), maximum(times_all[i,:,10^3])))

    push!(plots_reward, p)

    # Insert path and comment out for exporting figure as a pdf
    # save("INSERT_PATH/reward-$i.pdf", plots_reward[i])

    # Original line
    save("Mathematik/POMDPs/Coding/Julia/POMDPs/natural-policy-gradients/graphics/reward-$i.pdf", plots_reward[i])

end
