# Contributors Guide

Thank you for your interest in contributing to `EasyHybrid.jl`! We welcome everyone, whether you're new to open source, Julia, or scientific computing, or an experienced developer. Every contribution, big or small, helps make this project better.

If you have questions, ideas, or just want to chat, please reach out to us anytime. We're happy to help and discuss anything related to `EasyHybrid.jl` or science in general.

## How You Can Contribute

* **Report bugs or suggest features:** [Open a GitHub issue](https://github.com/EarthyScience/EasyHybrid.jl/issues/new/) to let us know about problems or ideas.
* **Start or join a discussion:** [Create a GitHub discussion](https://github.com/EarthyScience/EasyHybrid.jl/discussions/new/choose) to ask questions, share experiences, or brainstorm.
* **Improve documentation:** Help us make our docs clearer and more helpful for everyone.
* **Write code:** Fix bugs, add features, or improve performance.

We aim to follow the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative practices. We encourage you to read it before submitting a pull request, but don't worry! we're flexible and happy to help you through the process.

### Tips for Creating Issues

The most helpful bug reports:

* Include a clear code snippet (not just a link) that shows the problem in the latest version of `EasyHybrid.jl`. A ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example) is ideal.
* Paste the full error message you received, even if it's long.
* Use triple backticks (```` ``` ````) for code, and [markdown formatting](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to keep things readable.
* Share your `EasyHybrid.jl` version, Julia version, and details about your computer or environment.

Discussions are great for questions about usage, implementation, science, or anything else.

## Ready to Start Coding?

* Fork the [`EasyHybrid.jl` repository](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks), make your changes, and [open a pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). We'll review and help you get it merged.
* For small fixes (like typos), you can use the [GitHub editor](https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository) for a quick edit and pull request.
* Please try to follow our code style and formatting conventions. 

> [!TIP]
> We use the [Runic](https://github.com/fredrikekre/Runic.jl) style. To format your code, use the script at [/tools/formatter/](https://github.com/EarthyScience/EasyHybrid.jl/blob/main/tools/formatter/format.jl):
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
> Run this from `tools/formatter`, then review and commit your changes before making a pull request.

> [!NOTE]
> If you’re unsure about formatting, don’t worry! we’ll help you and can apply it for you.

## Good First Steps

* Try out `EasyHybrid.jl` using the examples in our documentation, or create your own. If you hit any problems or have questions, please open an issue!
* Write an example or tutorial showing how to use `EasyHybrid.jl` for something interesting.
* Suggest improvements to documentation or comments.
* Implement a new feature you’d like to see.

If you want to work on something, let us know by commenting on an issue or opening a new one. This helps us coordinate and support you.

We’re excited to have you join our community. Thank you for helping make `EasyHybrid.jl` better!