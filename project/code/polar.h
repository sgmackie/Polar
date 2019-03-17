#ifndef polar_h
#define polar_h

//Type defines
#include "polar_typedefs.h"

//TODO: Remove CRT
#include <stdlib.h>
#include <stdio.h>
#include <string.h> //memcpy
#include <time.h>
#include <stdarg.h>
#include "math.h"

/*                  */
/*  Global code  	*/
/*                  */


//Hex defines
//Sources
#define SO_NONE                 0x09070232
#define SO_OSCILLATOR           0x1c5903fe
#define SO_FILE                 0x08d10222

//Effects
#define FX_DRY                  0x06a701ed
#define FX_AM                   0x04af018c
#define FX_ECHO                 0x0898021d

//Envelopes
#define EN_NONE                 0x089f0223
#define EN_ADSR                 0x0861021d
#define EN_BREAKPOINT           0x1b0903e2
#define EN_AMPLITUDE            0x178a0398
#define EN_FREQUENCY            0x17ad03a5

//Random
#define RN_AMPLITUDE            0x182603a5
#define RN_FREQUENCY            0x184903b2
#define RN_PAN                  0x06b401df

//OSC messages
//Source types
#define LN_                     9201655152285363179U
#define SO_                     8744316438972908U

//Events
#define PLAY                    11120484276852016966U
#define FADE                    7677966677680727406U
#define VECTOR                  12143376858605269818U
#define MATRIX                  16755126490873392952U

//General Defines
#define Mono    1
#define Stereo  2
#define Quad    4

//Function macros
#define Hash(X) math_HashGenerate(X)
#define DB(X) math_DecibelToLinear(X)
#define AMP_MIN(X) DB(X)
#define AMP_MAX(X) DB(X)
#define AMP(X) DB(X)


/*                  */
/*  Math code  	    */
/*                  */

typedef struct VECTOR4D
{
    f32 X;
    f32 Y;
    f32 Z;
    f32 W;
} VECTOR4D;

typedef struct MATRIX_4x4
{
    f32 A1, A2, A3, A4;
    f32 B1, B2, B3, B4;
    f32 C1, C2, C3, C4;
    f32 D1, D2, D3, D4;
} MATRIX_4x4;



/*                  */
/*  Memory code  	*/
/*                  */

//Defines
//Define maximum alignemt Size fron integral types
typedef f64 max_align_type;
global_scope size_t MaxAlignment = alignof(max_align_type);
#define ARENA_DEFAULT_CHUNK_SIZE 2048

//Structs
//Chunks are a linked list of allocated data
typedef struct MEMORY_ARENA_CHUNK 
{
    size_t CurrentSize;
    size_t TotalSize;
    MEMORY_ARENA_CHUNK *NextChunk;
    char Data[sizeof(char)];
} MEMORY_ARENA_CHUNK;

//Arena to access all allocated chunks as a list
typedef struct MEMORY_ARENA 
{
    MEMORY_ARENA_CHUNK *FirstInList;
    size_t CurrentSize;
    size_t TotalSize;
    size_t ChunkCount;
    size_t UnalignedDataSize;
} MEMORY_ARENA;

//Protoypes
//Chunks
MEMORY_ARENA_CHUNK *memory_arena_ChunkCreate(size_t Size);                  //Create an aligned chunk at a given size by calling system alloctor (VirtualAlloc / mmap)
void memory_arena_ChunkDestroy(MEMORY_ARENA_CHUNK *Chunk);                  //Free any created chunk

//Arena
MEMORY_ARENA *memory_arena_Create(size_t Size);                             //Allocate space for the arena and create the first chunk in the linked list
void memory_arena_Destroy(MEMORY_ARENA *Arena);                             //Free arena and all linked chunks
void memory_arena_Reset(MEMORY_ARENA *Arena);                               //Call memset on every chunk in the arena
void *memory_arena_Push(MEMORY_ARENA *Arena, void *Type, size_t Size);      //Push any data onto the arena by assigning it a specific address from the chunks and return that address
void memory_arena_Pull(MEMORY_ARENA *Arena);                                //Free any chunks that are currently empty
void memory_arena_Print(MEMORY_ARENA *Arena, const char *Name);             //Print the contents of a given arena


/*                  */
/*  Buffer code  	*/
/*                  */

#define RINGBUFFER_DEFAULT_BLOCK_COUNT 3

//Structs
typedef struct POLAR_BUFFER
{
    u32 SampleCount;
    f32 *Data;
} POLAR_BUFFER;

typedef struct POLAR_RINGBUFFER
{
    u64 Samples;
    u64 ReadAddress;
    u64 WriteAddress;
    u64 TotalBlocks;
    u64 CurrentBlocks;
    i16 *Data;
} POLAR_RINGBUFFER;

//Prototypes
POLAR_RINGBUFFER *polar_ringbuffer_Create(MEMORY_ARENA *Arena, u32 Samples, u32 Blocks);        //Allocate buffer that is segemented into blocks of samples, default is 3
void polar_ringbuffer_Destroy(MEMORY_ARENA *Arena, POLAR_RINGBUFFER *Buffer);                   //Free buffer

//Writing
i16 *polar_ringbuffer_WriteData(POLAR_RINGBUFFER *Buffer);                                      //Return the address of the next available block to write to
bool polar_ringbuffer_WriteCheck(POLAR_RINGBUFFER *Buffer);                                     //Check that there is space to write another block of samples
void polar_ringbuffer_WriteFinish(POLAR_RINGBUFFER *Buffer);                                    //Change the address to point to the next block

//Reading
i16 *polar_ringbuffer_ReadData(POLAR_RINGBUFFER *Buffer);                                       //Read from the next available block
bool polar_ringbuffer_ReadCheck(POLAR_RINGBUFFER *Buffer);                                      //Check that there are any written blocks to read from
void polar_ringbuffer_ReadFinish(POLAR_RINGBUFFER *Buffer);                                     //Change the address to point to the next block


/*                  */
/*  Envelope code  	*/
/*                  */

//Defines
#define MAX_BREAKPOINT_LINE_LENGTH 64

//Structs
typedef struct POLAR_ENVELOPE_POINT
{
    f32 Time;
    f32 Value;
} POLAR_ENVELOPE_POINT;

typedef struct POLAR_ENVELOPE
{
    u32 Assignment;
    u32 CurrentPoints;
    u32 Index;
    POLAR_ENVELOPE_POINT Points[MAX_BREAKPOINTS];
} POLAR_ENVELOPE;


typedef struct POLAR_PER_SAMPLE_STATE
{
    f32 Current;
    f32 Previous;
    f32 StartValue;
    f32 EndValue;
    f32 StartTime;
    f32 Duration;
    bool IsFading;
} POLAR_PER_SAMPLE_STATE;



/*                  */
/*  DSP code        */
/*                  */

//Defines
#define PI32 3.14159265359f
#define TWO_PI32 (2.0 * PI32)

#define WV_SINE         0x094f023c
#define WV_SQUARE       0x0ee802de
#define WV_SAWDOWN      0x11e50330
#define WV_SAWUP        0x0bf6029d
#define WV_TRIANGLE     0x153e0363

//Structs
//Wave oscillator
typedef struct POLAR_OSCILLATOR
{
    u32 Waveform;                                   //Waveform assignment
    f32 (*Tick) (POLAR_OSCILLATOR *Oscillator);     //Function pointer to the different waveform ticks
    f32 TwoPiOverSampleRate;                        //2 * Pi / Sample rate is a constant variable
    f32 PhaseCurrent;
    f32 PhaseIncrement;                             //Store calculated phase increment
    POLAR_PER_SAMPLE_STATE Frequency;
    f64 FrequencyTarget;
    f64 FrequencyDelta;
} POLAR_OSCILLATOR;


//Prototypes
POLAR_OSCILLATOR *polar_dsp_OscillatorCreate(MEMORY_ARENA *Arena, u32 SampleRate, u32 WaveformSelect, f32 InitialFrequency);        //Create oscillator object
void polar_dsp_OscillatorInit(POLAR_OSCILLATOR *Oscillator, u32 SampleRate, u32 WaveformSelect, f32 InitialFrequency);              //Initialise elements of oscillator (can be used to reset)
f32 polar_dsp_PhaseWrap(f32 &Phase);                                                                                                //Wrap phase 2*Pi as precaution against sin(x) function on different compilers failing to wrap large scale values internally
f32 polar_dsp_TickSine(POLAR_OSCILLATOR *Oscillator);                                                                               //Calculate sine wave samples
f32 polar_dsp_TickSquare(POLAR_OSCILLATOR *Oscillator);                                                                             //Calculate square wave samples
f32 polar_dsp_TickSawDown(POLAR_OSCILLATOR *Oscillator);                                                                            //Calculate downward square wave samples
f32 polar_dsp_TickSawUp(POLAR_OSCILLATOR *Oscillator);                                                                              //Calculate upward square wave samples
f32 polar_dsp_TickTriangle(POLAR_OSCILLATOR *Oscillator);    

/*                  */
/*  File code  	    */
/*                  */

typedef struct POLAR_FILE
{
    u32 Channels;
    u32 SampleRate;
    u64 FrameCount;
    u64 ReadIndex;
    f32 *Samples;
} POLAR_FILE;

/*                  */
/*  Engine code  	*/
/*                  */

typedef struct POLAR_ENGINE         //Struct to hold platform specific audio API important engine properties
{
	u32 BufferSize;			    //Frame count for output buffer
    u32 LatencySamples;
	u16 Channels;                   //Engine current channels
    f32 *OutputChannelPositions;
	u32 SampleRate;                 //Engine current sampling rate
    u32 BytesPerSample;
    f32 UpdateRate;
    f32 NoiseFloor;                 //Attenuation noise floor
} POLAR_ENGINE;


//Prototypes
//String handling
i32 StringLength(const char *String);
i32 StringLength(const char *String)
{
    i32 Count = 0;

    while(*String++)
    {
        ++Count;
    }

    return Count;
}

void polar_StringConcatenate(size_t StringALength, const char *StringA, size_t StringBLength, const char *StringB, char *StringC);
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


/*                  */
/*  Sources code  	*/
/*                  */

//Structs
//Union to select between source types
typedef struct POLAR_SOURCE_TYPE
{
    u32 Flag;
    union
    {
        POLAR_OSCILLATOR *Oscillator;
        POLAR_FILE *File;
    };
} POLAR_SOURCE_TYPE;

typedef enum POLAR_SOURCE_PLAY_STATE
{
    Stopped,
    Stopping,
    Playing
} POLAR_SOURCE_PLAY_STATE;

//Current state of the source
typedef struct POLAR_SOURCE_STATE
{
    f32 *PanPositions;
    
    POLAR_PER_SAMPLE_STATE Amplitude;

    VECTOR4D Position;
    f32 MinDistance;
    f32 MaxDistance;
    f32 Rolloff;
    f32 RolloffFactor;
    bool RolloffDirty;
    bool IsDistanceAttenuated;

    u32 CurrentEnvelopes;
    POLAR_ENVELOPE Envelope[MAX_ENVELOPES];
} POLAR_SOURCE_STATE;

//Source is a struct of arrays that is accessed by it's unique ID
typedef struct POLAR_SOURCE
{
    u8 CurrentSources;
    u64 UID[MAX_SOURCES];
    POLAR_SOURCE_TYPE Type[MAX_SOURCES];
    POLAR_SOURCE_PLAY_STATE PlayState[MAX_SOURCES];
    POLAR_SOURCE_STATE States[MAX_SOURCES];
    u32 FX[MAX_SOURCES];
    u8 Channels[MAX_SOURCES];
    u32 SampleRate[MAX_SOURCES];
    u64 SampleCount[MAX_SOURCES];
    u32 BufferSize[MAX_SOURCES];
    f32 *Buffer[MAX_SOURCES];
} POLAR_SOURCE;


typedef struct POLAR_SOURCE_SOLO
{
    u64 UID;
    POLAR_SOURCE_TYPE Type;
    POLAR_SOURCE_PLAY_STATE PlayState;
    POLAR_SOURCE_STATE States;
    u32 FX;
    u8 Channels;
    u32 SampleRate;
    u64 SampleCount;
    u32 BufferSize;
    f32 *Buffer;
} POLAR_SOURCE_SOLO;



/*                  */
/*  Listener code   */
/*                  */

typedef struct POLAR_LISTENER
{
    u64 UID;
    VECTOR4D Position;
} POLAR_LISTENER;


/*                  */
/*  Mixer code      */
/*                  */

//Struct
//Container to hold a static amount of sources
typedef struct POLAR_CONTAINER
{
    u8 CurrentContainers;
    u64 UID[MAX_CONTAINERS];
    f64 Amplitude[MAX_CONTAINERS];
    u32 FX[MAX_CONTAINERS];
    POLAR_SOURCE Sources[MAX_CONTAINERS];
} POLAR_CONTAINER;

//Linked list of submixes with their own containers
typedef struct POLAR_SUBMIX
{
    u64 UID;
    f64 Amplitude;
    u32 FX;
    POLAR_CONTAINER Containers;
    u32 ChildSubmixCount;
    POLAR_SUBMIX *ChildSubmix;
    POLAR_SUBMIX *NextSubmix;
} POLAR_SUBMIX;

//Global mixer that holds all submixes and their child containers
typedef struct POLAR_MIXER
{
    f64 Amplitude;
    POLAR_LISTENER *Listener;
    u32 SubmixCount;
    POLAR_SUBMIX *FirstInList;
} POLAR_MIXER;

//Prototypes
//Mixer
POLAR_MIXER *polar_mixer_Create(MEMORY_ARENA *Arena, f64 Amplitude);            //Create mixing object to hold singly linked list of submixes
void polar_mixer_Destroy(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer);              //Free the mixer

//Submixing
void polar_mixer_SubmixCreate(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer, const char ParentUID[MAX_STRING_LENGTH], const char ChildUID[MAX_STRING_LENGTH], f64 Amplitude);     //Create a submix that is either assigned to any free space in the list or is the child of another submix
void polar_mixer_SubmixDestroy(POLAR_MIXER *Mixer, const char UID[MAX_STRING_LENGTH]);                                                                                      //Remove submix from the list

//Containers
void polar_mixer_ContainerCreate(POLAR_MIXER *Mixer, const char SubmixUID[MAX_STRING_LENGTH], const char ContainerUID[MAX_STRING_LENGTH], f64 Amplitude);                   //Create a container to hold any audio sources as a single group, then assign it to a submix
void polar_mixer_ContainerDestroy(POLAR_MIXER *Mixer, const char ContainerUID[MAX_STRING_LENGTH]);                                                                          //Remove container from the array

//Sources
// POLAR_SOURCE *polar_source_Retrieval(POLAR_MIXER *Mixer, u64 UID, u32 &SourceIndex);                                                              //Find a specific source and return it's struct with a given index
// void polar_source_Create(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer, POLAR_ENGINE Engine, const char ContainerUID[MAX_STRING_LENGTH], const char SourceUID[MAX_STRING_LENGTH], u32 Channels, u32 Type, ...);
// void polar_source_CreateFromFile(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer, POLAR_ENGINE Engine, const char *FileName);                                                       //Read CSV text file to create any sources
// void polar_source_Update(POLAR_MIXER *Mixer, POLAR_SOURCE *Sources, u32 &SourceIndex, f64 GlobalTime, f32 NoiseFloor);                                                                                                          //Internal update function used by polar_source_UpdatePlaying
// void polar_source_UpdatePlaying(POLAR_MIXER *Mixer, f64 GlobalTime, f32 NoiseFloor);                                                                                                                        //Update every playing source's current state
// void polar_source_Play(POLAR_MIXER *Mixer, u64 SourceUID, f64 GlobalTime, f32 Duration, f32 *PanPositions, u32 FX, u32 EnvelopeType, ...);                                          //Mark a source for playback
// void polar_source_Fade(POLAR_MIXER *Mixer, u64 SourceUID, f64 GlobalTime, f32 NewAmplitude, f32 Duration);                                                           //Change source amplitude with a fade over time in seconds

/*                  */
/*  Render code     */
/*                  */

//Prototypes

void polar_render_Source(u32 &SampleRate, u64 &SampleCount, u32 Samples, POLAR_SOURCE_TYPE &Type, u32 &FX, f32 *Buffer);     //Fills a source's buffer and applies any FX
void polar_render_SumStereo(POLAR_ENGINE PolarEngine, u8 &Channels, f32 *PanPositions, f64 &Amplitude, f32 *Buffer, f32 *SourceOutput);                                                 //Sums source to stereo output buffer
void polar_render_Container(POLAR_ENGINE PolarEngine, POLAR_SOURCE &ContainerSources, f64 ContainerAmplitude, f32 *ContainerOutput);                                                    //Render every source in a container and mix as a single buffer
void polar_render_Submix(POLAR_ENGINE PolarEngine, POLAR_SUBMIX *Submix, f32 *SubmixOutput);                                                                                            //Render every container in a submix
void polar_render_Callback(POLAR_ENGINE PolarEngine, POLAR_MIXER *Mixer, f32 *MixBuffer, i16 *MasterOutput);                                                                                  //Loop through each submix and render their containers/sources  


//Function pointers
void (*Summing)(POLAR_ENGINE, u8 &, f32 *, f32 &, f32 &, f32 *, f32 *);        //Pointer to summing function that is dependant on the output channel configuration
void (*Callback)(POLAR_ENGINE, POLAR_MIXER *, f32 *, i16 *);                   //Audio callback function called by the Windows/Linux audio API



#endif
