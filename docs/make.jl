using RandomFunctions
using Documenter

DocMeta.setdocmeta!(RandomFunctions, :DocTestSetup, :(using RandomFunctions); recursive=true)

makedocs(;
    modules=[RandomFunctions],
    authors="Šimon Soldát",
    sitename="RandomFunctions.jl",
    format=Documenter.HTML(;
        canonical="https://soldasim.github.io/RandomFunctions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/soldasim/RandomFunctions.jl",
    devbranch="main",
)
