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

    @testset "Type Stability (@inferred) & Zero-Allocation Hot Paths" begin
        exp32 = ExponentialProfile(10.0f0, 1.0f0, 0.5f0; FT = Float32)
        pl32 = PowerLawProfile(10.0f0, 1.0f0, 2.0f0; FT = Float32)
        ch32 = ChamberlainProfile(10.0f0, 1.0f0, 2.0f0; FT = Float32)
        @test (@inferred neutral_density(exp32, 1.5f0)) isa Float32
        @test (@inferred local_scale_height(exp32, 1.5f0)) ≈ 0.5f0
        @test (@inferred neutral_density(pl32, 2.0f0)) ≈ 2.5f0
        @test (@inferred neutral_density(ch32, 2.0f0)) ≈ 10.0f0 * exp(-1.0f0)
        @test (@inferred photoionization_rate(1.0f-6, 100.0f0)) isa Float32
        @test (@inferred charge_exchange_rate(2.0f-19, 5.0f0, 100.0f0, 4.0f5)) isa Float32

        for k in (0.0, 1.0, 2.0, 2.5, 3.0, 4.0)
            pk = PowerLawProfile(8.0, 1.0, k)
            @test neutral_density(pk, 2.0) ≈ 8.0 * 0.5^k
        end

        ep = exosphere_profile(; n0 = 4.0, r0 = 1.0, H = 0.5, center = (1.0, 0.0, 0.0))
        @test (@inferred ep(1.5)) isa Float64
        @test (@inferred ep(1.5, 0.2)) isa Float64
        @test (@inferred ep(1.5, 0.2, -0.1)) isa Float64

        c1 = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
        c2 = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
        c3 = NeutralComponentCode(:chamberlain, 50.0, 1.5, 0.0, 0.01, "He")
        sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0, half_height = 5.0)
        od_pl = OpticalDepth(; cross_sections = [0.05, 0.01, 0.02], chapman = false)
        od_ch = OpticalDepth(; cross_sections = [0.05, 0.01, 0.02], chapman = true)
        exo_sc = build_exosphere_model(; planet_radius = 1.0, components = [c1, c2, c3], shadow = sc, inner_radius = 1.0)
        exo_od = build_exosphere_model(; planet_radius = 1.0, components = [c1, c2, c3], optical_depth = od_pl, inner_radius = 1.0)
        exo_ch = build_exosphere_model(; planet_radius = 1.0, components = [c1, c2, c3], optical_depth = od_ch, inner_radius = 1.0)

        x_day = SVector(1.5, 0.0, 0.0)
        x_night = SVector(-1.5, 0.5, 0.0)
        x_graz = SVector(-1.5, 1.2, 0.0)
        x_term = SVector(-0.15, 1.5, 0.0)

        @test (@inferred shadow_factor(sc, x_night)) == 0.0
        @test (@inferred exo_component_density(exo_ch, 1.5, 1)) isa Float64
        @test (@inferred exo_density(exo_ch, 1.5)) isa Float64
        @test (@inferred exo_scale_height(exo_ch, 1.5, 1)) isa Float64
        @test (@inferred chapman_function(20.0, 0.7)) isa Float64
        @test (@inferred chapman_function(20.0, -0.1)) isa Float64
        @test (@inferred photo_attenuation(exo_sc, x_night, hypot(x_night...))) == 0.0
        @test (@inferred photo_attenuation(exo_od, x_day, 1.5)) isa Float64
        @test (@inferred photo_attenuation(exo_od, x_graz, hypot(x_graz...))) isa Float64
        @test (@inferred photo_attenuation(exo_ch, x_day, 1.5)) isa Float64
        @test (@inferred photo_attenuation(exo_ch, x_term, hypot(x_term...))) isa Float64

        sumN = zeros(3)
        sumNA = zeros(3)
        @test (@inferred integrate_exosphere_box!(sumN, sumNA, exo_ch, x_day, SVector(0.2, 0.2, 0.2), 0, 1.0)) isa Float64

        # Warmup and check 0 B heap allocations
        for f_call in (
                () -> shadow_factor(sc, x_night),
                () -> exo_component_density(exo_ch, 1.5, 2),
                () -> exo_density(exo_ch, 1.5),
                () -> exo_scale_height(exo_ch, 1.5, 3),
                () -> chapman_function(20.0, -0.1),
                () -> photo_attenuation(exo_sc, x_night, 1.5811388300841898),
                () -> photo_attenuation(exo_od, x_day, 1.5),
                () -> photo_attenuation(exo_od, x_graz, 1.9209372712298547),
                () -> photo_attenuation(exo_ch, x_day, 1.5),
                () -> photo_attenuation(exo_ch, x_term, 1.5074813431681335),
                () -> integrate_exosphere_box!(sumN, sumNA, exo_ch, x_day, SVector(0.2, 0.2, 0.2), 0, 1.0),
                () -> integrate_exosphere_box!(sumN, sumNA, exo_ch, SVector(1.05, 0.1, 0.0), SVector(0.4, 0.4, 0.4), 0, 1.0),
                () -> integrate_exosphere_box!(sumN, sumNA, exo_ch, SVector(1.05, 0.1, 0.0), SVector(0.4, 0.4, 0.0), 0, 1.0),
                () -> photoionization_rate(1.0e-6, 100.0, 0.85),
                () -> charge_exchange_rate(2.0e-19, 5.0, 100.0, 4.0e5),
            )
            f_call()
            @test (@allocated f_call()) == 0
        end
    end
end
