using Aqua
using Exosphere
using StaticArrays
using Test

@testset "Exosphere.jl" begin
    @testset "Code Quality & Hygiene (Aqua.jl)" begin
        Aqua.test_all(
            Exosphere;
            ambiguities = true,
            piracies = true,
            stale_deps = true,
            deps_compat = (check_extras = false,),
        )
    end

    @testset "Neutral Profiles (1D/2D/3D & Allocation-Free)" begin
        exp_p = ExponentialProfile(10.0, 1.0, 0.5)
        @test neutral_density(exp_p, 1.0) ≈ 10.0
        @test neutral_density(exp_p, 1.5) ≈ 10.0 * exp(-1.0)
        @test local_scale_height(exp_p, 2.0) ≈ 0.5
        @test (@allocated neutral_density(exp_p, 1.5)) == 0

        pl_p = PowerLawProfile(10.0, 1.0, 2.0)
        @test neutral_density(pl_p, 2.0) ≈ 2.5
        @test local_scale_height(pl_p, 2.0) ≈ 1.0
        @test (@allocated neutral_density(pl_p, 2.0)) == 0

        ch_p = ChamberlainProfile(10.0, 1.0, 2.0)
        @test neutral_density(ch_p, 2.0) ≈ 10.0 * exp(-1.0)
        @test local_scale_height(ch_p, 2.0) ≈ 2.0
        @test (@allocated neutral_density(ch_p, 2.0)) == 0

        ep = exosphere_profile(; n0 = 4.0, r0 = 1.0, H = 0.5, center = (1.0, 0.0, 0.0))
        @test ep(1.0) ≈ 4.0
        @test ep(2.5, 0.0) ≈ 4.0 * exp(-1.0)
        @test ep(1.0, 0.3, 0.4) ≈ 4.0
        @test (@allocated ep(1.5, 0.2, -0.1)) == 0

        haser = exosphere_profile(; n0 = 9.0, r0 = 1.0, H = Inf, power = 2.0)
        @test haser(3.0, 0.0, 0.0) ≈ 1.0
    end

    @testset "ShadowCylinder, OpticalDepth, Chapman & Box Quadrature" begin
        sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0, half_height = 5.0)
        @test shadow_factor(sc, 2.0, 0.0, 0.0) == 1.0
        @test shadow_factor(sc, -2.0, 0.5, 0.0) == 0.0
        @test shadow_factor(sc, -2.0, 1.5, 0.0) == 1.0
        @test shadow_factor(sc, -6.0, 0.5, 0.0) == 1.0

        c1 = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
        c2 = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
        od = OpticalDepth(; cross_sections = [0.05, 0.0], solar_dir = (1.0, 0.0, 0.0), chapman = true)
        exo = build_exosphere_model(;
            planet_radius = 1.0, components = [c1, c2], optical_depth = od, inner_radius = 1.0,
        )
        @test n_components(exo) == 2
        @test exo_component_density(exo, 0.9, 1) == 0.0
        @test exo_density(exo, 1.0) ≈ 110.0

        @test chapman_function(20.0, 1.0) ≈ 1.0 atol = 0.05
        att_day = photo_attenuation(exo, SVector(1.5, 0.0, 0.0), 1.5)
        att_night = photo_attenuation(exo, SVector(-1.5, 0.0, 0.0), 1.5)
        @test 0.0 < att_night < att_day <= 1.0

        sumN = zeros(2)
        sumNA = zeros(2)
        vol = integrate_exosphere_box!(sumN, sumNA, exo, SVector(1.5, 0.0, 0.0), SVector(0.2, 0.2, 0.2), 0, 1.0)
        @test vol ≈ 0.008
        @test sumN[1] > 0.0 && sumNA[1] > 0.0 && sumNA[1] <= sumN[1]

        @test photoionization_rate(2.0, 3.0, 0.5) ≈ 3.0
        @test charge_exchange_rate(0.1, 2.0, 3.0, 4.0) ≈ 2.4
    end
end
