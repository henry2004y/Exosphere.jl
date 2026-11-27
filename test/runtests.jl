using Aqua: Aqua
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
using Exosphere
using StaticArrays: SVector
using Test

function _alloc_neutral_density(p, r)
    neutral_density(p, r)
    return @allocated neutral_density(p, r)
end

function _alloc_exosphere_profile(ep, x, y, z)
    ep(x, y, z)
    return @allocated ep(x, y, z)
end

function _alloc_shadow_factor(sc, x)
    shadow_factor(sc, x)
    return @allocated shadow_factor(sc, x)
end

function _alloc_exo_component_density(exo, r, c)
    exo_component_density(exo, r, c)
    return @allocated exo_component_density(exo, r, c)
end

function _alloc_exo_density(exo, r)
    exo_density(exo, r)
    return @allocated exo_density(exo, r)
end

function _alloc_exo_scale_height(exo, r, c)
    exo_scale_height(exo, r, c)
    return @allocated exo_scale_height(exo, r, c)
end

function _alloc_chapman(xp, cos_sza)
    chapman_function(xp, cos_sza)
    return @allocated chapman_function(xp, cos_sza)
end

function _alloc_photo_attenuation(exo, x, r)
    photo_attenuation(exo, x, r)
    return @allocated photo_attenuation(exo, x, r)
end

function _alloc_integrate_box!(sumN, sumNA, exo, c, h, depth, rcut)
    integrate_exosphere_box!(sumN, sumNA, exo, c, h, depth, rcut)
    return @allocated integrate_exosphere_box!(sumN, sumNA, exo, c, h, depth, rcut)
end

function _alloc_photoionization_rate(nu, n, att)
    photoionization_rate(nu, n, att)
    return @allocated photoionization_rate(nu, n, att)
end

function _alloc_charge_exchange_rate(sig, ni, nn, v)
    charge_exchange_rate(sig, ni, nn, v)
    return @allocated charge_exchange_rate(sig, ni, nn, v)
end

@testset "Exosphere.jl" begin
    @testset "Package Hygiene, Aqua.jl & API Docstrings" begin
        Aqua.test_all(
            Exosphere;
            ambiguities = true,
            piracies = true,
            stale_deps = true,
            deps_compat = (check_extras = (ignore = [:Test],),),
        )
        @test check_no_implicit_imports(Exosphere; skip = (Base, Core)) === nothing
        @test check_no_stale_explicit_imports(Exosphere) === nothing

        docs_meta = Base.Docs.meta(Exosphere)
        missing_docstrings = [
            s for s in names(Exosphere) if
                s !== :Exosphere &&
                !haskey(docs_meta, Base.Docs.Binding(Exosphere, s))
        ]
        @test isempty(missing_docstrings)
    end

    @testset "Validation & Edge Cases" begin
        # Profile constructor validation
        @test_throws ArgumentError ExponentialProfile(-1.0, 1.0, 0.5)
        @test_throws ArgumentError ExponentialProfile(Inf, 1.0, 0.5)
        @test_throws ArgumentError ExponentialProfile(1.0, 0.0, 0.5)
        @test_throws ArgumentError ExponentialProfile(1.0, 1.0, 0.0)

        @test_throws ArgumentError PowerLawProfile(-1.0, 1.0, 2.0)
        @test_throws ArgumentError PowerLawProfile(1.0, 0.0, 2.0)
        @test_throws ArgumentError PowerLawProfile(1.0, 1.0, -0.5)

        @test_throws ArgumentError ChamberlainProfile(-1.0, 1.0, 2.0)
        @test_throws ArgumentError ChamberlainProfile(1.0, 0.0, 2.0)
        @test_throws ArgumentError ChamberlainProfile(1.0, 1.0, Inf)

        @test_throws ArgumentError exosphere_profile(; n0 = -1.0)
        @test_throws ArgumentError exosphere_profile(; r0 = 0.0)
        @test_throws ArgumentError exosphere_profile(; H = 0.0)
        @test_throws ArgumentError exosphere_profile(; power = -1.0)
        @test_throws ArgumentError exosphere_profile(; center = (NaN, 0.0, 0.0))

        # Clamping at r <= r0 and H = Inf exponential limit
        exp_inf = ExponentialProfile(5.0, 1.0, Inf)
        @test neutral_density(exp_inf, 0.5) == 5.0
        @test neutral_density(exp_inf, 3.0) == 5.0
        @test local_scale_height(exp_inf, 2.0) == 0.0

        pl = PowerLawProfile(7.0, 1.5, 2.0)
        @test neutral_density(pl, 0.8) == 7.0
        ch = ChamberlainProfile(7.0, 1.5, 2.0)
        @test neutral_density(ch, 0.8) == 7.0
        ep = exosphere_profile(; n0 = 4.0, r0 = 1.5, H = 0.5, power = 2.0)
        @test ep(1.0) == 4.0

        # NeutralComponentCode, ShadowCylinder, OpticalDepth & build_exosphere_model validation
        @test_throws ArgumentError NeutralComponentCode(:unknown, 1.0, 0.5, 0.0, 1.0)
        @test_throws ArgumentError NeutralComponentCode(:exponential, 1.0, 0.0, 0.0, 1.0, "O")
        @test_throws ArgumentError NeutralComponentCode(:chamberlain, 1.0, 0.0, 0.0, 1.0, "He")
        @test_throws ArgumentError NeutralComponentCode(:powerlaw, 1.0, 0.0, 2.0, 0.0, "H")

        @test_throws ArgumentError ShadowCylinder(; solar_dir = (0.0, 0.0, 0.0))
        @test_throws ArgumentError ShadowCylinder(; solar_dir = [1.0, 0.0])

        @test_throws ArgumentError OpticalDepth(; cross_sections = [-0.1])
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.0, 0.0])
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], scale_heights = [1.0, 2.0])
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], solar_dir = (0.0, 0.0, 0.0))
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], cos_sza_floor = 0.0)
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], tau_floor = -1.0)
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], tau_cutoff = 0.0)
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], min_production = 1.5)

        c1 = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
        c_zero_k = NeutralComponentCode(:powerlaw, 10.0, 0.0, 0.0, 0.01, "Flat")
        sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0)
        od = OpticalDepth(; cross_sections = [0.05], chapman = true)
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = [c1], shadow = sc, optical_depth = od,
        )
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = NeutralComponentCode[],
        )
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = [c1, c_zero_k], optical_depth = od,
        )
        od_zero_h = OpticalDepth(; cross_sections = [0.05])
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = [c_zero_k], optical_depth = od_zero_h,
        )

        # Unattenuated ExosphereModel & BoundsError
        exo_plain = build_exosphere_model(; planet_radius = 1.0, components = [c1])
        @test photo_attenuation(exo_plain, SVector(-2.0, 0.0, 0.0), 2.0) == 1.0
        @test_throws BoundsError exo_scale_height(exo_plain, 1.5, 0)
        @test_throws BoundsError exo_scale_height(exo_plain, 1.5, 2)

        # Explicit scale_heights & chapman_component override
        od_custom = OpticalDepth(;
            cross_sections = [0.05, 0.01],
            scale_heights = [0.4, nothing],
            chapman = true,
            chapman_component = 2,
        )
        c2 = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
        exo_custom = build_exosphere_model(;
            planet_radius = 1.0, components = [c1, c2], optical_depth = od_custom,
        )
        @test exo_scale_height(exo_custom, 2.0, 1) ≈ 0.4
        @test exo_scale_height(exo_custom, 2.0, 2) ≈ 1.0
        @test exo_custom.optical.chapman_component == 2

        # Non-Chapman grazing shadow, tau_cutoff, and chapman_tau_max branches
        od_cut = OpticalDepth(;
            cross_sections = [5.0],
            chapman = false,
            min_production = 1.0e-5,
            tau_cutoff = 2.0,
        )
        exo_cut = build_exosphere_model(;
            planet_radius = 1.0, components = [c1], optical_depth = od_cut,
        )
        @test photo_attenuation(exo_cut, SVector(-1.5, 0.5, 0.0), hypot(-1.5, 0.5)) == 1.0e-5
        @test photo_attenuation(exo_cut, SVector(1.0, 0.0, 0.0), 1.0) == 1.0e-5
        @test 1.0e-5 < photo_attenuation(exo_cut, SVector(-1.5, 2.3, 0.0), hypot(-1.5, 2.3)) < 1.0

        od_tmax = OpticalDepth(;
            cross_sections = [1.0],
            chapman = true,
            min_production = 1.0e-4,
            chapman_tau_max = 0.5,
        )
        exo_tmax = build_exosphere_model(;
            planet_radius = 1.0, components = [c1], optical_depth = od_tmax,
        )
        @test photo_attenuation(exo_tmax, SVector(1.0, 0.0, 0.0), 1.0) ≈ 1.0e-4

        # Chapman function branches (y < 8, 8 <= y < 100, y >= 100 for dayside & nightside)
        @test chapman_function(20.0, 0.7) > 1.0
        @test 0.0 < chapman_function(200.0, 1.0) < 2.0
        @test chapman_function(25000.0, 1.0) == 0.0
        @test chapman_function(20.0, -0.1) > 1.0
        @test chapman_function(200.0, -0.9) > 0.0
        @test chapman_function(25000.0, -0.95) == 0.0

        # Box quadrature: fully interior, 2D collapsed axis, and analytic sinh(b)/b accuracy
        sumN = zeros(1)
        sumNA = zeros(1)
        @test integrate_exosphere_box!(
            sumN, sumNA, exo_plain, SVector(0.2, 0.0, 0.0), SVector(0.2, 0.2, 0.2), 0, 1.0,
        ) == 0.0
        @test iszero(sumN) && iszero(sumNA)

        fill!(sumN, 0.0)
        fill!(sumNA, 0.0)
        area_straddle = integrate_exosphere_box!(
            sumN, sumNA, exo_plain, SVector(1.0, 0.0, 0.0), SVector(0.4, 0.4, 0.0), 0, 1.0,
        )
        @test 0.0 < area_straddle < 0.16
        @test sumN[1] > 0.0

        c_wide = NeutralComponentCode(:exponential, 10.0, 1.0, 0.0, 0.01, "O")
        exo_wide = build_exosphere_model(; planet_radius = 1.0, components = [c_wide])
        fill!(sumN, 0.0)
        fill!(sumNA, 0.0)
        dx, dy, dz = 0.2, 1.0e-5, 1.0e-5
        x0 = 2.0
        vol_exact = integrate_exosphere_box!(
            sumN, sumNA, exo_wide, SVector(x0, 0.0, 0.0), SVector(dx, dy, dz), 0, 1.0,
        )
        @test vol_exact ≈ dx * dy * dz
        b = 0.5 * dx / 1.0
        n_exact = 10.0 * exp(-(x0 - 1.0) / 1.0) * (sinh(b) / b) * dx * dy * dz
        @test isapprox(sumN[1], n_exact; rtol = 1.0e-9)
    end

    @testset "Neutral Profiles (1D/2D/3D & Allocation-Free)" begin
        exp_p = ExponentialProfile(10.0, 1.0, 0.5)
        @test neutral_density(exp_p, 1.0) ≈ 10.0
        @test neutral_density(exp_p, 1.5) ≈ 10.0 * exp(-1.0)
        @test local_scale_height(exp_p, 2.0) ≈ 0.5
        @test _alloc_neutral_density(exp_p, 1.5) == 0

        pl_p = PowerLawProfile(10.0, 1.0, 2.0)
        @test neutral_density(pl_p, 2.0) ≈ 2.5
        @test local_scale_height(pl_p, 2.0) ≈ 1.0
        @test _alloc_neutral_density(pl_p, 2.0) == 0

        ch_p = ChamberlainProfile(10.0, 1.0, 2.0)
        @test neutral_density(ch_p, 2.0) ≈ 10.0 * exp(-1.0)
        @test local_scale_height(ch_p, 2.0) ≈ 2.0
        @test _alloc_neutral_density(ch_p, 2.0) == 0

        ep = exosphere_profile(; n0 = 4.0, r0 = 1.0, H = 0.5, center = (1.0, 0.0, 0.0))
        @test ep(1.0) ≈ 4.0
        @test ep(2.5, 0.0) ≈ 4.0 * exp(-1.0)
        @test ep(1.0, 0.3, 0.4) ≈ 4.0
        @test _alloc_exosphere_profile(ep, 1.5, 0.2, -0.1) == 0

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

        @test _alloc_shadow_factor(sc, x_night) == 0
        @test _alloc_exo_component_density(exo_ch, 1.5, 2) == 0
        @test _alloc_exo_density(exo_ch, 1.5) == 0
        @test _alloc_exo_scale_height(exo_ch, 1.5, 3) == 0
        @test _alloc_chapman(20.0, -0.1) == 0
        @test _alloc_photo_attenuation(exo_sc, x_night, hypot(x_night...)) == 0
        @test _alloc_photo_attenuation(exo_od, x_day, 1.5) == 0
        @test _alloc_photo_attenuation(exo_od, x_graz, hypot(x_graz...)) == 0
        @test _alloc_photo_attenuation(exo_ch, x_day, 1.5) == 0
        @test _alloc_photo_attenuation(exo_ch, x_term, hypot(x_term...)) == 0
        @test _alloc_integrate_box!(sumN, sumNA, exo_ch, x_day, SVector(0.2, 0.2, 0.2), 0, 1.0) == 0
        @test _alloc_integrate_box!(sumN, sumNA, exo_ch, SVector(1.05, 0.1, 0.0), SVector(0.4, 0.4, 0.4), 0, 1.0) == 0
        @test _alloc_integrate_box!(sumN, sumNA, exo_ch, SVector(1.05, 0.1, 0.0), SVector(0.4, 0.4, 0.0), 0, 1.0) == 0
        @test _alloc_photoionization_rate(1.0e-6, 100.0, 0.85) == 0
        @test _alloc_charge_exchange_rate(2.0e-19, 5.0, 100.0, 4.0e5) == 0
    end
end
