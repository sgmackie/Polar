#ifndef polar_h
#define polar_h

//Type defines
#include "polar_typedefs.h"

//TODO: Remove CRT
#include <stdlib.h>
#include <stdio.h>
#include <string.h> //memcpy


//TODO: Finish rest of the comments
/*                  */
/*  General code  	*/
/*                  */

//Structs
typedef struct POLAR_INPUT_STATE
{
    i32 HalfTransitionCount;
    bool EndedDown;
} POLAR_INPUT_STATE;

typedef struct POLAR_INPUT_CONTROLLER
{
    bool IsConnected;

    union State
    {
        POLAR_INPUT_STATE Buttons[12];
        
		struct Press
        {
            POLAR_INPUT_STATE MoveUp;
            POLAR_INPUT_STATE MoveDown;
            POLAR_INPUT_STATE MoveLeft;
            POLAR_INPUT_STATE MoveRight;
            
            POLAR_INPUT_STATE ActionUp;
            POLAR_INPUT_STATE ActionDown;
            POLAR_INPUT_STATE ActionLeft;
            POLAR_INPUT_STATE ActionRight;
            
            POLAR_INPUT_STATE LeftShoulder;
            POLAR_INPUT_STATE RightShoulder;

        	POLAR_INPUT_STATE Back;
        	POLAR_INPUT_STATE Start;
            
            POLAR_INPUT_STATE Terminator;
        } Press;
    } State;

} POLAR_INPUT_CONTROLLER;


typedef struct POLAR_INPUT
{
    POLAR_INPUT_STATE MouseButtons[5];
    i32 MouseX;
	i32 MouseY;
	i32 MouseZ;
 
    POLAR_INPUT_CONTROLLER Controllers[5];
} POLAR_INPUT;


//BYTE buffer passed to WASAPI for rendering
typedef struct POLAR_BUFFER
{
	u32 FramePadding;
	u32 FramesAvailable;
	f32 *SampleBuffer;
	void *DeviceBuffer;
} POLAR_BUFFER;

typedef struct POLAR_DATA       //Struct to hold platform specific audio API important engine properties
{
	POLAR_BUFFER Buffer;       	//Float and device buffers for rendering
	u32 BufferFrames;			//Frame count for output buffer
	u16 Channels;               //Engine current channels
	u32 SampleRate;             //Engine current sampling rate
	u16 BitRate;                //Engine current bitrate
} POLAR_DATA;


//Prototypes
//String handling
void polar_StringConcatenate(size_t StringALength, const char *StringA, size_t StringBLength, const char *StringB, char *StringC);
i32 polar_StringLengthGet(const char *String);

//Input handling
POLAR_INPUT_CONTROLLER *ControllerGet(POLAR_INPUT *Input, u32 ControllerIndex);


//TODO: Move these functions to a .cpp file
//Iterate through strings A & B and push to string C
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


i32 polar_StringLengthGet(const char *String)
{
    i32 Count = 0;

    while(*String++)
    {
        ++Count;
    }

    return Count;
}



POLAR_INPUT_CONTROLLER *ControllerGet(POLAR_INPUT *Input, u32 ControllerIndex)
{
    Assert(ControllerIndex < ArrayCount(Input->Controllers));
    
    POLAR_INPUT_CONTROLLER *Result = &Input->Controllers[ControllerIndex];
    return Result;
}


/*                  */
/*  Memory code  	*/
/*                  */


typedef struct POLAR_MEMORY
{
    bool IsInitialized;

    u64 PermanentDataSize;
    void *PermanentData;

    u64 TemporaryDataSize;
    void *TemporaryData;
} POLAR_MEMORY;


/*                  */
/*  File code       */
/*                  */

//Defines
//64 bit max size
//TODO: Check on x86 builds
#define WAV_FILE_MAX_SIZE  ((u64)0xFFFFFFFFFFFFFFFF)

typedef struct POLAR_WAV_HEADER //WAV file specification "http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html"
{
	u16 AudioFormat;		    //1 for WAVE_FORMAT_PCM, 3 for WAVE_FORMAT_IEEE_FLOAT
	u16 NumChannels;		    //2
	u32 SampleRate;			    //192000
	u32 ByteRate;			    //SampleRate * NumChannels * BitsPerSample/8
	u16 BlockAlign;			    //NumChannels * BitsPerSample/8
	u16 BitsPerSample;		    //32
	u64 DataChunkDataSize;	    //Overall size of the "data" chunk
	u64 DataChunkDataStart;	    //Starting byte of the data chunk
} POLAR_WAV_HEADER;

typedef struct POLAR_WAV
{
	FILE *WAVFile;              //Handle to a file
	const char *Path;			//Path to a file
	POLAR_WAV_HEADER WAVHeader; //Struct to store WAV header properties
    //TODO: Support i16/i32 data
	f32 *Data;                  //Floating point sample buffer
	u64 TotalSampleCount;       //Total samples in a file when read
} POLAR_WAV;

//Protypes
//File writing
POLAR_WAV *polar_render_WAVWriteCreate(const char *FilePath, POLAR_DATA *Engine);
void polar_render_WAVWriteDestroy(POLAR_WAV *File);
internal bool polar_render_WAVWriteHeader(POLAR_WAV *File, POLAR_DATA *Engine);
internal size_t polar_render_WAVWriteRaw(POLAR_WAV *File, size_t BytesToWrite, const void *FileData);
internal u64 polar_render_WAVWriteFloat(POLAR_WAV *File, u64 SamplesToWrite, const void *FileData);
internal u32 polar_render_RIFFChunkRound(u64 RIFFChunkSize);
internal u32 polar_render_DataChunkRound(u64 DataChunkSize);

/*                  */
/*  DSP code        */
/*                  */

#include "math.h"

#define PI32 3.14159265359f
#define TWO_PI32 (2.0 * PI32)

//Structs
//Waveform select
typedef enum WAVEFORM
{
    SINE,
    SQUARE,
    SAWDOWN,
    SAWUP,
    TRIANGLE
} WAVEFORM;

//Wave oscillator
typedef struct POLAR_OSCILLATOR
{
    WAVEFORM Waveform;
    f64 (*Tick) (POLAR_OSCILLATOR *Oscillator);   //Function pointer to the different waveform ticks
    f64 TwoPiOverSampleRate;                //2 * Pi / Sample rate is a constant variable
    f64 PhaseCurrent;
    f64 PhaseIncrement;                     //Store calculated phase increment
    f64 FrequencyCurrent;
} POLAR_OSCILLATOR;


//Prototypes
POLAR_OSCILLATOR *polar_wave_OscillatorCreate(u32 SampleRate, WAVEFORM WaveformSelect, f64 InitialFrequency);                                                                                //Allocation and initialisation functions in one
void polar_wave_OscillatorDestroy(POLAR_OSCILLATOR *Oscillator);                                                                //Free oscillator struct
void polar_wave_OscillatorInit(POLAR_OSCILLATOR *Oscillator, u32 SampleRate, WAVEFORM WaveformSelect, f64 InitialFrequency);    //Initialise elements of oscillator (can be used to reset)
f64 polar_wave_PhaseWrap(f64 &Phase);                                                                                     //Wrap phase 2*Pi as precaution against sin(x) function on different compilers failing to wrap large scale values internally
f64 polar_wave_TickSine(POLAR_OSCILLATOR *Oscillator);                                                                          //Calculate sine wave samples
f64 polar_wave_TickSquare(POLAR_OSCILLATOR *Oscillator);                                                                        //Calculate square wave samples
f64 polar_wave_TickSawDown(POLAR_OSCILLATOR *Oscillator);                                                                       //Calculate downward square wave samples
f64 polar_wave_TickSawUp(POLAR_OSCILLATOR *Oscillator);                                                                         //Calculate upward square wave samples
f64 polar_wave_TickTriangle(POLAR_OSCILLATOR *Oscillator);    


/*                  */
/*  Object code  	*/
/*                  */

typedef struct POLAR_OBJECT_STATE
{
	f32 Frequency;
	f32 Amplitude;
	f32 Pan;
    WAVEFORM Waveform;
} POLAR_OBJECT_STATE;

typedef struct POLAR_OBJECT
{
    u32 UID;
    char Name[64];
    POLAR_OSCILLATOR *Oscillator;
    POLAR_OBJECT_STATE *State;
} POLAR_OBJECT;

typedef struct POLAR_OBJECT_ARRAY 
{
    u32 Count;
    POLAR_OBJECT **Objects;

    POLAR_OBJECT_STATE *States;
    u32 StateCount;

    POLAR_OSCILLATOR *Oscillators;
    u32 OscillatorCount;

} POLAR_OBJECT_ARRAY;

/*                  */
/*  Rendering code  */
/*                  */

//Prototypes
//Rendering
internal f32 polar_render_PanPositionGet(u16 Position, f32 Amplitude, f32 PanFactor);    //Calculate stereo pan position
internal void polar_render_BufferFill(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, void *DeviceBuffer, f32 *FileSamples, POLAR_OSCILLATOR *Osc, POLAR_OBJECT_STATE *State);

//Create function pointer for rendering callback (an external function loaded dynamically)
#define POLAR_RENDER_CALLBACK(FunctionName) void FunctionName(POLAR_DATA &Engine, POLAR_WAV *File, POLAR_OBJECT_ARRAY *Array, POLAR_MEMORY *Memory, POLAR_INPUT *Input)
typedef POLAR_RENDER_CALLBACK(polar_render_Update);




POLAR_OSCILLATOR *polar_wave_OscillatorNextInArray(POLAR_OSCILLATOR *Array, u32 *Count);

POLAR_OSCILLATOR *polar_wave_OscillatorNextInArray(POLAR_OSCILLATOR *Array, u32 *Count)
{
    POLAR_OSCILLATOR *Result = &Array[*Count];
    Count += 1;
    return Result;
}


POLAR_OBJECT_STATE *polar_object_StateNextInArray(POLAR_OBJECT_STATE *Array, u32 *Count);

POLAR_OBJECT_STATE *polar_object_StateNextInArray(POLAR_OBJECT_STATE *Array, u32 *Count)
{
    POLAR_OBJECT_STATE *Result = &Array[*Count];
    Count += 1;
    return Result;
}






#endif
