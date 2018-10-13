# Polar

- [About](#about)
- [Changelog](#changelog)

### About: <a name="about"></a>

Simple real-time audio engine in C/C++. Currently using WASAPI for Windows playback with plans to expand to MacOS (CoreAudio).

Primary use is as a test bed for any DSP projects I'm currently working on and to experiment with game audio concepts related to audio engine design.

### Changelog: <a name="changelog"></a>

- v0.2 (Current):
    - Polar:
        - Began to seperate the rendering streams & threads away from the WASAPI layer to be more platform independent
        - Created the POLAR_BUFFER object that is used to write and read audio data
    
- v0.1:
    - WASAPI:
        - Created platform initial layer (multi-threaded)
    - Polar:
        - Created basic interface functions to the WASAPI layer
        - Created POLAR_OBJECTS that are used to store audio data