# Contributors Guide

Thank you for considering contributing to `EasyHybrid.jl`! There are multiple ways to contribute, and we appreciate all contributions.

Feel free to ask us questions and chat with us at any time about any topic at all.

## Ways to Contribute

* [Opening a GitHub issue](https://github.com/EarthyScience/EasyHybrid.jl/issues/new/)

* [Creating a GitHub discussion](https://github.com/EarthyScience/EasyHybrid.jl/discussions/new/choose)

We aim at following the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices, although at the moment we are not very strict about it so that there is room for everyone.

We ask that new contributors read that guide before submitting a pull request.

### Creating issues

The simplest way to contribute to `EasyHybrid.jl` is to create or comment on issues and discussions.

The most useful bug reports:

* Provide an explicit code snippet -- not just a link -- that reproduces the bug in the latest tagged version of `EasyHybrid.jl`. This is sometimes called the ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example). Reducing bug-producing code to a minimal example can dramatically decrease the time it takes to resolve an issue.

* Paste the _entire_ error received when running the code snippet, even if it's unbelievably long.

* Use triple backticks (```` ``` ````) to enclose code snippets, and other [markdown formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to make your issue easy and quick to read.

* Report the `EasyHybrid.jl` version, Julia version, machine and any other possibly useful details of the computational environment in which the bug was created.

Discussions are recommended for asking questions about (for example) the user interface, implementation details, science, and life in general.

## But I want to _code_!

* New users can help write `EasyHybrid.jl` code and documentation by [forking the `EasyHybrid.jl` repository](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks), [using git](https://guides.github.com/introduction/git-handbook/) to edit code and docs, and then creating a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Pull requests are reviewed by `EasyHybrid.jl` collaborators.

* A pull request can be merged once it is reviewed and approved by collaborators. If the pull request author has write access, they have the responsibility of merging their pull request. Otherwise, `EasyHybrid.jl` collaborators will execute the merge with permission from the pull request author.

* Note: for small or minor changes (such as fixing a typo in documentation), the [GitHub editor](https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository) is super useful for forking and opening a pull request with a single click.

* Write your code with love and care. In particular, conform to existing `EasyHybrid.jl` style and formatting conventions. 

> [!TIP]
> For formatting decisions we follow the [Runic](https://github.com/fredrikekre/Runic.jl) style. We have prepare a ready to use script at [/tools/formatter/](https://github.com/EarthyScience/EasyHybrid.jl/blob/main/tools/formatter/format.jl).
>
> ```julia
> using Pkg
> Pkg.activate(@__DIR__)
> Pkg.instantiate()
> using Runic
>
> dir = joinpath(@__DIR__, "..", "..")
> Runic.main(["--verbose", "--inplace", dir])
> ```
> run the project from `tools/formatter`, then review `add / commit / push` those changes before moving forward with a pull request.

> [!NOTE]
> Don't worry if you don't follow this last step, we will still move forward with your contributions and apply the formatting later on.

## What's a good way to start contributing to `EasyHybrid.jl`?

* Try to run `EasyHybrid.jl` and play around with it with the examples from the documentation, or create your own! If you run into any bugs/problems or find it difficult to use or understand, please open an issue!

* Write up an example or tutorial on how to do something useful with `EasyHybrid.jl`.

* Improve documentation or comments if you found something hard to use.

* Implement a new feature if you need it to use `EasyHybrid.jl`.

If you're interested in working on something, let us know by commenting on
existing issues or by opening a new issue. This is to make sure no one else
is working on the same issue and so we can help and guide you in case there
is anything you need to know beforehand.