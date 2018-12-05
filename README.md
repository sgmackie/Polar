# Polar

- [About](#about)
- [Changelog](#changelog)

### About: <a name="about"></a>

Simple real-time audio engine in C/C++. Currently using WASAPI for Windows playback with plans to expand to MacOS (Core Audio).

Primary use is as a test bed for any DSP projects I'm currently working on and to experiment with game audio concepts related to audio engine design.

### Changelog: <a name="changelog"></a>
    
- v0.1 (Current, rolled back from v0.2):
    - Polar:
        - Preliminary POLAR_DATA struct to hold platform audio API data and critical audio device properties (channels / sampling rate)
    - Windows:
        - Create Window to process input messages and display debug information
        - Added performance timing counters
            - WASAPI:
                - Re-wrote WASAPI implementation (now included in source)