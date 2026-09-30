using NeuralPDE, SciMLBase, Lux, Random, Test
using AdvancedHMC, MCMCChains, LogDensityProblems

@testset "Bayesian ODE HMC adaptation draws" begin
    prob = ODEProblem((u, p, t) -> cos(2π * t), 0.0, (0.0, 1.0))
    chain = Lux.Chain(Lux.Dense(1, 2, tanh), Lux.Dense(2, 1))
    n_adapted(stats) = count(s.is_adapt for s in stats)

    Random.seed!(100)
    _, _, stats_default = ahmc_bayesian_pinn_ode(prob, chain; draw_samples = 30)
    @test n_adapted(stats_default) == 3

    Random.seed!(100)
    _, samples, stats = ahmc_bayesian_pinn_ode(
        prob, chain; draw_samples = 30, n_adapts = 20
    )
    @test n_adapted(stats) == 20
    @test length(samples) == 30

    @test_throws ArgumentError ahmc_bayesian_pinn_ode(
        prob, chain; draw_samples = 30, n_adapts = 31
    )

    Random.seed!(100)
    alg = BNNODE(chain; draw_samples = 30, numensemble = 4, n_adapts = 20)
    @test alg.n_adapts == 20
    sol = solve(prob, alg)
    @test n_adapted(sol.original.statistics) == 20
end
