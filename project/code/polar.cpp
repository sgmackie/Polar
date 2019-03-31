//Polar
#include "polar.h"

i32 StringLength(const char *String)
{
    i32 Count = 0;

    while(*String++)
    {
        ++Count;
    }

    return Count;
}

void polar_StringConcatenate(size_t StringALength, const char *StringA, size_t StringBLength, const char *StringB, char *StringC)
{
    for(u32 Index = 0; Index < StringALength; ++Index)
    {
        *StringC++ = *StringA++;
    }

    for(u32 Index = 0; Index < StringBLength; ++Index)
    {
        *StringC++ = *StringB++;
    }

    *StringC++ = 0;
}


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
#include "polar_GUI.cpp"