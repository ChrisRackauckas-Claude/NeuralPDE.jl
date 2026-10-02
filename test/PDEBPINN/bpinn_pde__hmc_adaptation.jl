using NeuralPDE, ModelingToolkit, SciMLBase, Lux, Random, Test
using AdvancedHMC, MCMCChains, LogDensityProblems
using DomainSets: Interval

@testset "Bayesian PDE HMC adaptation draws" begin
    @parameters t
    @variables u(..)
    Dt = Differential(t)
    @named sys = PDESystem(
        Dt(u(t)) ~ 1.0, [u(0.0) ~ 0.0],
        [t ∈ Interval(0.0, 1.0)], [t], [u(t)]
    )
    disc = BayesianPINN(
        Lux.Chain(Lux.Dense(1, 1)), GridTraining(0.5); rng = Xoshiro(100)
    )
    n_adapted(sol) = count(s.is_adapt for s in sol.original.statistics)

    Random.seed!(100)
    sol_default = ahmc_bayesian_pinn_pde(
        sys, disc; draw_samples = 30, numensemble = 4, pretrain_iters = 0, saveats = [0.5]
    )
    @test n_adapted(sol_default) == 3

    Random.seed!(100)
    sol = ahmc_bayesian_pinn_pde(
        sys, disc; draw_samples = 30, numensemble = 4, pretrain_iters = 0, saveats = [0.5],
        n_adapts = 20
    )
    @test n_adapted(sol) == 20
    @test length(sol.original.samples) == 30

    @test_throws ArgumentError ahmc_bayesian_pinn_pde(
        sys, disc; draw_samples = 30, numensemble = 4, saveats = [0.5], n_adapts = 31
    )
end
