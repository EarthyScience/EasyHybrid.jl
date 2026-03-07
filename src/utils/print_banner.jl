export print_banner

const DOC_URL = "https://earthyscience.github.io/EasyHybrid.jl/"

const EASY = [
    "  ______                ",
    " |  ____|               ",
    " | |__   __ _ ___ _   _ ",
    " |  __| / _` / __| | | |",
    " | |___| (_| \\__ \\ |_| |",
    " |______\\__,_|___/\\__, |",
    "                   __/ |",
    "                  |___/ ",
]

const HYBRID = [
    " _    _       _          _     _  ",
    "| |  | |     | |        (_)   | | ",
    "| |__| |_   _| |__  _ __| | __| | ",
    "|  __  | | | | '_ \\| '__| |/ _` | ",
    "| |  | | |_| | |_) | |  | | (_| | ",
    "|_|  |_|\\__, |_.__/|_|  |_|\\__,_| ",
    "         __/ |                    ",
    "        |___/                     ",
]

function print_banner(; version_string = _get_version_string())

    cols = displaysize(stdout)[2]

    if cols < 110
        println("EasyHybrid $version_string\n$DOC_URL\nType \"?EasyHybrid\" for more information.")
        return
    end

    info = [
        "",
        " |  Simple & flexible framework for hybrid modeling",
        " |  Integrating neural networks with process-based models",
        " |",
        " |  Type \"?EasyHybrid\" for more information.",
        " |",
        " |  Version $version_string",
        " |  $DOC_URL",
    ]

    use_color = get(stdout, :color, false)
    for i in eachindex(EASY)
        if use_color
            print(EASY[i])
            printstyled(HYBRID[i], color = :red)
        else
            print(EASY[i])
            print(HYBRID[i])
        end
        println(info[i])
    end

    return
end

const EASYHYBRID_VERSION = Base.pkgversion(@__MODULE__)

function _get_version_string()
    version_string = string(EASYHYBRID_VERSION)
    if isdir(joinpath(@__DIR__, "..", ".git"))
        # try to get the current git commit date
        try
            date = readchomp(`git -C $(joinpath(@__DIR__, "..")) log -1 --format=%cd --date=short`)
            version_string *= "-dev ($date)"
        catch
            version_string *= "-dev"
        end
    else
        # get the date from the git tag for this version, if available
        try
            tag = "v$version_string"
            date = readchomp(`git -C $(joinpath(@__DIR__, "..")) log -1 --format=%cd --date=short $tag`)
            isempty(date) || (version_string *= " ($date)")
        catch
            # not a git repo or no matching tag, no date appended
        end
    end
    return version_string
end

function __init__()
    return if isinteractive() && get(ENV, "EASYHYBRID_PRINT_BANNER", "true") != "false"
        print_banner()
    end
end
