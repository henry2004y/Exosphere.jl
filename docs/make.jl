using Documenter
using DocumenterVitepress
using Exosphere

include(joinpath(@__DIR__, "..", "examples", "planetary_exosphere_euv.jl"))

run_planetary_exosphere_euv(;
    outpath = joinpath(@__DIR__, "src", "assets", "planetary_exosphere_euv.png"),
)

DocMeta.setdocmeta!(
    Exosphere,
    :DocTestSetup,
    :(using Exosphere, StaticArrays);
    recursive = true,
)

makedocs(;
    modules = [Exosphere],
    authors = "Hongyang Zhou <hyzhou@umich.edu> and contributors",
    repo = Remotes.GitHub("henry2004y", "Exosphere.jl"),
    sitename = "Exosphere.jl",
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/henry2004y/Exosphere.jl",
        devbranch = "main",
        devurl = "dev",
        build_vitepress = get(ENV, "BUILD_VITEPRESS", "true") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "getting_started.md",
            "Profiles & Attenuation" => "physics.md",
            "Sources & Quadrature" => "quadrature_coupling.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
    warnonly = false,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/henry2004y/Exosphere.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
