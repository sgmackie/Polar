# Polar

- [About](#about)
- [Changelog](#changelog)
- [Acknowledgments](#acknowledgments)

### About: <a name="about"></a>

Simple real-time audio engine in C/C++. Currently using WASAPI for Windows playback with plans to expand to MacOS (Core Audio).

Primary use is as a test bed for any DSP projects I'm currently working on and to experiment with game audio concepts related to audio engine design.

### Changelog: <a name="changelog"></a>

- v0.2 (Current):
    - Polar:
        - Added basic WAV file rendering (only tested with the current audio device properties)
- v0.1 (Rolled back from previous WASAPI build):
    - Polar:
        - Preliminary POLAR_DATA struct to hold platform audio API data and critical audio device properties (channels / sampling rate)
        - Stereo panning from a float value
    - Windows:
        - Create Window to process input messages and display debug information
        - Added performance timing counters
            - WASAPI:
                - Re-wrote WASAPI implementation (now included in source)

### Acknowledgments: <a name="acknowledgments"></a>

- Platform code:
    - Handmade Hero (https://handmadehero.org/)
- File code:
    - dr_wav (https://mackron.github.io/)