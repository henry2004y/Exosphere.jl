Format Julia code with [Runic.jl](https://github.com/fredrikekre/Runic.jl) (92-char limit; enforced by [`.github/workflows/Format.yml`](./.github/workflows/Format.yml)).

# Code and commits
- Avoid `Any` in function signatures, and use an explicit `;` before keyword arguments in calls.
- Keep hot-path kernels type-stable and allocation-free; add `@inferred` and `@allocated == 0` checks in [`test/runtests.jl`](./test/runtests.jl) for new hot-path APIs.
- Document every `export` and `public` symbol (checked by [`test/runtests.jl`](./test/runtests.jl) and `checkdocs = :exports` in [`docs/make.jl`](./docs/make.jl)); add internal comments or docstrings only when behavior is not obvious from names and types.
- Commit titles follow `<type>: brief summary` (e.g., `feat:`, `fix:`, `perf:`, `docs:`, `style:`, `chore:`) with a brief prose body when needed; link external issues/PRs as `owner/repo#123`.
- Respect manual user edits to code you wrote; explain concerns and ask before reintroducing modified or deleted code.

# Dependencies
- Manage dependencies with `Pkg.add` and `Pkg.rm`; edit [`Project.toml`](./Project.toml) directly only for `[compat]`.
- Keep runtime dependencies minimal and use explicit imports (`using Pkg: sym`) in `src/`; [`test/runtests.jl`](./test/runtests.jl) enforces `Aqua.jl` and `ExplicitImports.jl`.
- [`docs/Project.toml`](./docs/Project.toml) and [`benchmark/Project.toml`](./benchmark/Project.toml) wire the local package via `[sources]`.

# Testing, docs, and benchmarks
- Full test suite (unit, `Aqua.jl`, `ExplicitImports.jl`, docstring, `@inferred`, and zero-allocation checks): `julia --project=. -e 'using Pkg; Pkg.test()'`
- Docs build (set `BUILD_VITEPRESS=false` to skip VitePress while still running `@example` blocks and `@docs` checks): `julia --project=docs docs/make.jl`
- Benchmarks: `julia --project=benchmark -e 'using BenchmarkTools; include("benchmark/benchmarks.jl"); run(SUITE)'` (or `benchpkg Exosphere --rev=main,dirty` via `AirspeedVelocity.jl`).
- Update [`docs/src/api.md`](./docs/src/api.md) (and [`README.md`](./README.md) when appropriate) whenever `export` or `public` names change.
