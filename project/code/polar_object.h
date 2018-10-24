#ifndef polar_object_h
#define polar_object_h

enum POLAR_OBJECT_TYPE
{
    NONE,
    PLR_OSC,
    PLR_WAV,
};

enum PLR_OSC_WAVEFORM
{
    PLR_SINE,
    PLR_SQUARE,
    PLR_SAWDOWN,
    PLR_SAWUP,
    PLR_TRIANGLE,
};

typedef struct POLAR_OBJECT
{
    u64 ObjectID;
    char *ObjectName;
    POLAR_OBJECT_TYPE ObjectType;
    char *ObjectTypeName;
    
    union
    {
        OSCILLATOR *WaveOscillator;
        POLAR_BUFFER *FileBuffer;
    };

    RENDER_STREAM *StreamHandle;

    //TODO: Linked list references to next/previous objects?
} POLAR_OBJECT;

typedef struct POLAR_OBJECT_ARRAY
{
    u32 Size;
    POLAR_OBJECT **Objects;
} POLAR_OBJECT_ARRAY;

#include "array.cpp"

POLAR_OBJECT polar_object_CreateObject(u64 ID, char *Name, POLAR_OBJECT_TYPE Type);
void polar_object_DestroyObject(POLAR_OBJECT Object);

POLAR_OBJECT_ARRAY polar_object_CreateObjectArray(u32 Size);

//TODO: How do to pass flags for optional arguments? VarArgs?
void polar_object_SubmitObject(POLAR_OBJECT &Object, RENDER_STREAM *Stream, PLR_OSC_WAVEFORM Flags);

#endif