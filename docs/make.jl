using LearningPi
using Documenter
using Literate

DocMeta.setdocmeta!(LearningPi, :DocTestSetup, :(using LearningPi); recursive = true)

makedocs(;
    modules = [LearningPi],
    authors = "F. Demelas, M. Lacroix, J. Le Roux, A. Parmentier",
    repo = "/-/blob/main/src/LearningPi.jl",
    sitename = "LearningPi.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/FDemelas/Learning_Lagrangian_Multipliers.jl/-/blob/main/src/Learning_Lagrangian_Multipliers.jl",
        assets = String[]),
    pages = [
        "Home" => "index.md",
        "Loss Functions" => "Loss.md",
	"Machine Learning Models" => "Models.md",
	"Sampling Mechanism" => "Sampling.md", 
    "Training Scripts" => "Training.md", 
        "API reference" => "api.md"
    ]
)

deploydocs(;
    repo = "https://github.com/FDemelas/Learning_Lagrangian_Multipliers.jl",
    devbranch = "main"
)
