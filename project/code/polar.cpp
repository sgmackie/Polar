//Polar
#include "polar.h"


f64 polar_WallTime()
{
#ifdef _WIN32
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;

#elif __linux__
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}

//Source
#include "polar_math.cpp"
#include "polar_memory.cpp"
#include "polar_buffer.cpp"
#include "polar_envelope.cpp"
#include "polar_dsp.cpp"
#include "polar_effect.cpp"
#include "polar_mixer.cpp"
#include "polar_listener.cpp"
#include "polar_source.cpp"
#include "polar_OSC.cpp"
#include "polar_render.cpp"