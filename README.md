# Polar

- [About](#about)
- [Build](#build)
- [Changelog](#changelog)
- [Acknowledegments](#acknowledgements)

### About: <a name="about"></a>

Simple real-time audio engine in C/C++. Currently supports Windows (WASAPI) and Linux (ALSA), with a port to macOS planned (CoreAudio).

Primary use is as a test bed for any DSP projects I'm currently working on and to experiment with game audio concepts related to engine design.

### Build: <a name="build"></a>

All platforms were built using the current Clang toolset (https://releases.llvm.org/download.html)

- Windows (x64):
    - VS2017: Call vcvars64.bat from the Visual Studio install directory and run build.bat (or build_MSVC.bat if Clang is not installed)
- Linux (x64):
    - Clang: Run build.sh

### Changelog: <a name="changelog"></a>

- v0.2 (Current):
    - Linux:
        - Started ALSA implementation
    - Windows:
        - Able to record input and object states and loop them back
- v0.1 (Rolled back from previous WASAPI build):
    - Polar:
        - Preliminary POLAR_DATA struct to hold platform audio API data and critical audio device properties (channels / sampling rate)
        - Stereo panning from a float value
    - Windows:
        - Create Window to process input messages and display debug information
        - Added performance timing counters
            - WASAPI:
                - Re-wrote WASAPI implementation (now included in source)

### Acknowledegments: <a name="acknowledgements"></a>

- Platform code:
    - Handmade Hero (https://handmadehero.org/)
- File code:
    - dr_wav (https://mackron.github.io/)
- Misc:
    - Using "ctime" for compilation metrics (https://gist.github.com/mmozeiko/4972c28be07e2b7c10dc88089eec7083)