#ifndef polar_object_h
#define polar_object_h

typedef enum struct POLAR_OBJECT_TYPE
{
    PLR_OSC,
    PLR_WAV,
} POLAR_OBJECT_TYPE;

typedef enum struct PLR_OSC_WAVEFORM
{
    SINE,
    SQUARE,
} PLR_OSC_WAVEFORM;

typedef struct POLAR_OBJECT
{
    u64 ObjectID;
    char *ObjectName;
    POLAR_OBJECT_TYPE ObjectType;
    char *ObjectTypeName;
    union
    {
        OSCILLATOR *WaveOscillator;
    };

    //TODO: Double pointer to the render stream?
    RENDER_STREAM *StreamHandle;
} POLAR_OBJECT;

POLAR_OBJECT polar_object_CreateObject(u64 ID, char *Name, POLAR_OBJECT_TYPE Type);
void polar_object_DestroyObject(POLAR_OBJECT Object);
//TODO: How do to pass flags for optional arguments?
void polar_object_SubmitObject(POLAR_OBJECT &Object, RENDER_STREAM *Stream, PLR_OSC_WAVEFORM Flags);

#endif