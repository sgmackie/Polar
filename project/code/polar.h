#ifndef polar_h
#define polar_h

//CRT
#include <stdlib.h>

//Type defines
#include "../external/polar_typedefs.h"

//Includes
//Synthesis
#include "../external/entropy/entropy.h"



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

//TODO: Check if Clang compliant, may need to name structs
typedef struct POLAR_INPUT_CONTROLLER
{
    bool IsConnected;

    union
    {
        POLAR_INPUT_STATE Buttons[12];
        
		struct
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
        };
    };
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
void polar_StringConcatenate(size_t StringALength, char *StringA, size_t StringBLength, char *StringB, char *StringC);
i32 polar_StringLengthGet(char *String);

//Input handling
POLAR_INPUT_CONTROLLER *ControllerGet(POLAR_INPUT *Input, u32 ControllerIndex);


//TODO: Move these functions to a .cpp file
//Iterate through strings A & B and push to string C
void polar_StringConcatenate(size_t StringALength, char *StringA, size_t StringBLength, char *StringB, char *StringC)
{
    for(i32 Index = 0; Index < StringALength; ++Index)
    {
        *StringC++ = *StringA++;
    }

    for(i32 Index = 0; Index < StringBLength; ++Index)
    {
        *StringC++ = *StringB++;
    }

    *StringC++ = 0;
}


i32 polar_StringLengthGet(char *String)
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
/*  Object code  	*/
/*                  */

typedef struct POLAR_OBJECT_STATE
{
	f32 Frequency;
	f32 Amplitude;
	f32 Pan;
} POLAR_OBJECT_STATE;


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
/*  Rendering code  */
/*                  */

//Prototypes
//Rendering
internal f32 polar_render_PanPositionGet(u16 Position, f32 Amplitude, f32 PanFactor);    //Calculate stereo pan position
internal void polar_render_BufferFill(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, void *DeviceBuffer, f32 *FileSamples, OSCILLATOR *Osc, POLAR_OBJECT_STATE *State);

//Create function pointer for rendering callback (an external function loaded dynamically)
#define POLAR_RENDER_CALLBACK(FunctionName) void FunctionName(POLAR_DATA &Engine, POLAR_WAV *File, OSCILLATOR *Osc, POLAR_MEMORY *Memory, POLAR_INPUT *Input)
typedef POLAR_RENDER_CALLBACK(polar_render_Update);


#endif