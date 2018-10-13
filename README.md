# Polar:

### About:

Simple real-time audio engine in C/C++. Currently using WASAPI for Windows playback.

Primary use is as a test platform for any DSP projects I'm currently working on.

### Changelog:

- v0.5 (Current):
    - Polar:
        - Began to seperate the rendering streams & threads away from the WASAPI layer to be more platform independent
        - Created the POLAR_BUFFER object that is used to write and read audio data
    
- v0.1:
    - WASAPI:
        - Created platform initial layer (multi-threaded)
    
    - Polar:
        - Created basic interface functions to the WASAPI layer
        - Created POLAR_OBJECTS that are used to store audio data