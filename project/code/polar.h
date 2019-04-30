//Headers
#include "core/core.h"
#include "core/log.cpp"

//Define name of logger here for macro functions
static LOGGER Logger = {};
bool core_CreateLogger(char const *Path, int Level, bool IsQuiet)
{
    FILE *LogFile = 0;
#ifdef _WIN32        
    fopen_s(&LogFile, Path, "w+");
#else       
    LogFile = fopen(Path, "w+");
#endif        

    if(!LogFile)
    {
        printf("Failed to open file: %s\n", Path);
        return false;
    }

    Logger.Init(LogFile, Level, IsQuiet);
    return true;
}

void core_DestroyLogger()
{
    fclose(Logger.File);
}

//Assertion with logging
static bool AssertQuit = true;
static i64 AssertFailuresCounter = 0;

//Assertion function with optional logging
//If AssertQuit is set, exists program
#define Assert(Expression, ...) do {                            \
  if(!(Expression))                                             \
  {                                                             \
    ++AssertFailuresCounter;                                    \
    Logger.Log(LOG_FATAL,    __FILE__, __LINE__, __VA_ARGS__);  \
    if(AssertQuit)                                              \
    {                                                           \
        abort();                                                \
    }                                                           \
  }                                                             \
} while(0);                                                     \

//CUDA
#if CUDA
#include "polar_cuda.h"
#include "cuda/polar_cuda_error.h"
#endif

//External functions
#include "../external/xxhash.c"     //Fast hasing function
#define FastHash(Name) XXH64(Name, strlen(Name), 0)
#include "../external/csv_read.c"
#include "../external/split.c"
#include "../external/pcg_basic.c"  //Random number generator
#define DR_WAV_IMPLEMENTATION
#include "../external/dr_wav.h"

#include "../external/udp.hh"
#include "../external/oscpkt.hh"

//IMGUI
#include "../external/imgui/imgui.cpp"
#include "../external/imgui/imgui_widgets.cpp"
#include "../external/imgui/imgui_draw.cpp"
#include "../external/imgui/imgui_demo.cpp"

//Core functions
#include "core/functions.cpp"       //General functions (alignment checks)
#include "core/memory/arena.cpp"    //Arena creation (alloc, resize, freeall)
#include "core/memory/pool.cpp"     //Pool creation (alloc, free, freeall)
#include "core/math.cpp"            //Math functions
// #include "core/string.cpp"          //String handling

//Max sizes
#define MAX_SOURCES         256
#define MAX_VOICES          16
#define MAX_STRING_LENGTH   128
#define MAX_BUFFER_SIZE     8192
#define MAX_BREAKPOINTS     64

typedef struct CMP_DURATION
{
    //Data    
    u64         SampleCount;
    u32         FrameDelay;
} CMP_DURATION;

typedef struct CMP_FORMAT
{
    //Data    
    u32 SampleRate;
    u32 Channels;

    //Functions
    void Init(u32 InputRate, u32 InputChannels);

} CMP_FORMAT;

typedef struct CMP_POSITION
{
    //Data    
    f32 X;
    f32 Y;
    f32 Z;
    f32 W;
} CMP_POSITION;


typedef struct CMP_WAV
{
    f32     *Data;
    u8      BitRate;
    u32     SampleRate;
    u32     Channels;
    u64     Length;
    u64     ReadIndex;

    //Functions
    void Init(char const *Name);

} CMP_WAV;


//TODO: Create generic void * buffers to support any type
typedef struct CMP_BUFFER
{
    //Data
    size_t  Count;
    f32     *Data;

    //Functions
    void CreateFromArena(MEMORY_ARENA *Arena, size_t Type, size_t InputCount);
    void CreateFromPool(MEMORY_POOL *Pool, size_t InputCount);
    void Destroy();
    f32 *Write();
    f32 *Read();

} CMP_BUFFER;


#define RINGBUFFER_DEFAULT_BLOCK_COUNT 3
typedef struct CMP_RINGBUFFER
{
    //Data
    size_t  Count;
    size_t  ReadAddress;
    size_t  WriteAddress;
    size_t  TotalBlocks;
    size_t  CurrentBlocks;
    i16     *Data;

    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Type, size_t InputCount, size_t Blocks);
    void Destroy();
    i16 *Write();
    bool CanWrite();
    void FinishWrite();
    i16 *Read();
    bool CanRead();
    void FinishRead();

} CMP_RINGBUFFER;

typedef struct CMP_FADE
{
    //Data
    f64     Current;
    f64     Previous;
    f64     StartValue;
    f64     EndValue;
    f64     StartTime;
    f64     Duration;
    bool    IsFading;

    //Functions
    void Init(f64 Amplitude);

} CMP_FADE;


typedef struct CMP_PARAMETER
{
    //Flags
    typedef enum TYPE
    {
        LINEAR          = 1 << 0,
        LOGARITHMIC     = 1 << 1,
    } TYPE;

    //Data
    i32 Type;
    f64 CurrentValue;
	f64 StartValue;
	f64 EndValue;
	f64 StartTime;
	f64 DeltaTime;
	bool IsDone;

    void Init(f64 Value = 0, i32 Flag = CMP_PARAMETER::TYPE::LINEAR)
    {
        Type            |= Flag;
        CurrentValue    = Value;
		StartValue      = CurrentValue;
		EndValue        = CurrentValue;
		StartTime       = 0;
		DeltaTime       = 0;
		IsDone          = true;
    }

} CMP_PARAMETER;


typedef struct CMP_PAN
{
    //Flags
    typedef enum MODE
    {
        MONO        = 1 << 0,
        STEREO      = 1 << 1,
        WIDE        = 1 << 2,

    } MODE;

    //Data
    i32             Flag;
    f64             Amplitude;

    //Functions
    void Init(i32 Mode, f64 Pan);

} CMP_PAN;

typedef struct CMP_OSCILLATOR
{
    //Flags
    typedef enum TYPE
    {
        SQUARE      = 1 << 0,
        SINE        = 1 << 1,
        TRIANGLE    = 1 << 2,
        SAWTOOTH    = 1 << 3,
    } TYPE;

    //Data
    i32         Flag;
    f64         Phasor;
    f64         PhaseIncrement;
    f64         SizeOverSampleRate;
    f64         Frequency;

    //Functions
    void Init(i32 Type, u32 SampleRate, f64 InputFrequency, f64 Limit = TWO_PI32);

} CMP_OSCILLATOR;


typedef struct CMP_NOISE
{
    //Flags
    typedef enum TYPE
    {
        WHITE   = 1 << 0,
        BROWN   = 1 << 1,
    } TYPE;

    //Data
    i32         Flag;
    f64         Accumulator;

    //Functions
    void Init(i32 Type);

} CMP_NOISE;


typedef struct CMP_ADSR
{
    bool IsActive;
    f64 MaxAmplitude;
    f64 Attack;
    f64 Decay;
    f64 SustainAmplitude;
    f64 Release;
    u64 Index;
    u64 DurationInSamples;
} CMP_ADSR;

typedef struct CMP_BREAKPOINT_POINT
{
    f64 Value;
    f64 Time;
} CMP_BREAKPOINT_POINT;

typedef struct CMP_BREAKPOINT
{
    //Data
    u64 Index;
    size_t Count;
    CMP_BREAKPOINT_POINT *Points;

    //Functions
    void Init(size_t PointIndex, f64 InputValue, f64 InputTime);
    void CreateFromArena(MEMORY_ARENA *Arena, size_t Type, size_t InputCount);
    void CreateFromPool(MEMORY_POOL *Pool, size_t InputCount);
    void Destroy();

} CMP_BREAKPOINT;


typedef struct CMP_MODULATOR
{
    //Flags
    typedef enum ASSIGNMENT
    {
        AMPLITUDE       = 1 << 0,
        FREQUENCY       = 1 << 1,
    } ASSIGNMENT;

    typedef enum TYPE
    {
        LFO_OSCILLATOR  = 1 << 2,
        ENV_ADSR        = 1 << 3,
        ENV_BREAKPOINT  = 1 << 4,
    } TYPE;

    i32 Flag;
    union
    {
        CMP_OSCILLATOR  Oscillator;
        CMP_ADSR        ADSR;
        CMP_BREAKPOINT  Breakpoint;
    };

    void Init(i32 Type, i32 Assignment)
    {
        Flag |= Type;
        Flag |= Assignment;
    }

} CMP_MODULATOR;


typedef struct CMP_BIQUAD
{
    typedef enum TYPE
    {
        LOWPASS     = 1 << 0,
        HIGHPASS    = 1 << 1,
    } TYPE;

    //Coefficients 
    f64 A0;
    f64 A1;
    f64 A2;
    
    f64 B1;
    f64 B2;

    f64 C0;
    f64 D0;

    //Delay
    f64 DelayInput1;
    f64 DelayInput2;
    f64 DelayOutput1;
    f64 DelayOutput2;

} CMP_BIQUAD;


typedef u64 ID_SOURCE;
typedef u64 ID_VOICE;
typedef struct HANDLE_SOURCE
{
    ID_SOURCE   ID;
    size_t      Index;
} HANDLE_SOURCE;

typedef struct HANDLE_VOICE
{
    ID_VOICE    ID;
    size_t      Index;
} HANDLE_VOICE;

typedef struct CMP_VOICEMAP
{
    //!Make dynamic
    HANDLE_VOICE Handles[MAX_VOICES];
    size_t Count;
} CMP_VOICEMAP;

typedef union TPL_TYPE
{
    CMP_OSCILLATOR  Oscillator;
    CMP_NOISE       Noise;
    CMP_WAV         WAV;
} TPL_TYPE;

typedef struct ENTITY_SOURCES
{
    //Flags
    enum FLAG_COMPONENTS
    {
        PLAYBACK                = 1 << 0,
        AMPLITUDE               = 1 << 1,
        PAN                     = 1 << 2,
        POSITION                = 1 << 3,
        OSCILLATOR              = 1 << 4,
        NOISE                   = 1 << 5,
        WAV                     = 1 << 6,
        ADSR                    = 1 << 7,
        BREAKPOINT              = 1 << 8,
        MODULATOR               = 1 << 9,
    };

    //Data
    size_t                      Count;
    char                        **Names;
    ID_SOURCE                   *IDs;
    CMP_VOICEMAP                *Voices;
    CMP_FORMAT                  *Formats;
    CMP_FADE                    *Amplitudes;
    CMP_PARAMETER               *Amps;
    CMP_PAN                     *Pans;
    CMP_POSITION                *Positions;
    TPL_TYPE                    *Types;
    CMP_ADSR                    *ADSRs;
    CMP_BREAKPOINT              *Breakpoints;
    CMP_MODULATOR               *Modulators;
    i32                         *Flags;

    //Functions
    void Create                 (MEMORY_ARENA *Arena, size_t Size);
    void Destroy                (MEMORY_ARENA *Arena);
    void Init                   (size_t Index);
    ID_SOURCE AddByName         (MEMORY_POOL *Pool, char *Name);
    HANDLE_SOURCE AddByHash     (u64 Hash);
    bool Remove                 (MEMORY_POOL *Pool, ID_SOURCE ID);
    size_t RetrieveIndex        (ID_SOURCE ID);
    ID_SOURCE RetrieveID        (size_t Index);
    HANDLE_SOURCE RetrieveHandle(ID_SOURCE ID);

} ENTITY_SOURCES;

typedef enum STATE_PLAY
{
    STOPPED,
    STOPPING,
    PLAYING,
    PAUSED,
} STATE_PLAY;

typedef enum STATE_VOICE
{
    INACTIVE,
    ACTIVE,
    SPAWNING,
} STATE_VOICE;

typedef struct CMP_STATE
{
    STATE_VOICE Voice;
    STATE_PLAY Play;
} CMP_STATE;


typedef struct TPL_PLAYBACK
{
    //Data
    CMP_FORMAT      Format;
    CMP_BUFFER      Buffer;
    CMP_DURATION    Duration;
} TPL_PLAYBACK;


typedef struct ENTITY_VOICES
{
    //Data
    size_t              Count;
    ID_VOICE            *IDs;
    TPL_PLAYBACK        *Playbacks;
    CMP_STATE         *States;
    HANDLE_SOURCE       *Sources;
    TPL_TYPE            *Types;
    CMP_FADE            *Amplitudes;
    CMP_PAN             *Pans;
    CMP_BREAKPOINT      *Breakpoints;
    CMP_ADSR            *ADSRs;
    i32                 *Flags;

    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Size);
    void Destroy(MEMORY_ARENA *Arena);
    void Init(size_t Index);
    ID_VOICE Add(HANDLE_SOURCE Source);
    ID_VOICE RetrieveID(size_t Index);
    size_t RetrieveIndex(ID_VOICE ID);

} ENTITY_VOICES;



typedef struct SYS_PLAY
{
    //Data
    size_t          SystemCount;    
    ID_VOICE        *SystemVoices;
    
    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Size);
    void Destroy(MEMORY_ARENA *Arena);
    void Add(ID_VOICE ID);
    bool Remove(ID_VOICE ID);
    bool Start(ENTITY_VOICES *Voices, ID_VOICE ID, f64 InputDuration, u32 Delay = 0, bool IsAligned = true);
    void Update(ENTITY_VOICES *Voices, f64 Time, u32 PreviousSamplesWritten, u32 SamplesToWrite);

} SYS_PLAY;


typedef struct SYS_FADE
{
    //Data
    size_t          SystemCount;    
    ID_VOICE        *SystemVoices;
    
    //Functions
    void Create     (MEMORY_ARENA *Arena, size_t Size);
    void Destroy    (MEMORY_ARENA *Arena);
    void Add        (ID_VOICE ID);
    bool Remove     (ID_VOICE ID);
    bool Start      (ENTITY_VOICES *Voices, ID_VOICE ID, f64 Time, f64 Amplitude, f64 Duration);
    void Update     (ENTITY_VOICES *Voices, f64 Time);

} SYS_FADE;


typedef struct SYS_OSCILLATOR_SINE
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBufferWithModulation(CMP_OSCILLATOR &Oscillator, CMP_MODULATOR &Modulator, CMP_BUFFER &Buffer, size_t BufferCount);
    void RenderToBuffer     (CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_OSCILLATOR_SINE;



typedef struct SYS_OSCILLATOR_SQUARE
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_OSCILLATOR_SQUARE;


typedef struct SYS_OSCILLATOR_TRIANGLE
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_OSCILLATOR_TRIANGLE;


typedef struct SYS_OSCILLATOR_SAWTOOTH
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (CMP_OSCILLATOR &Oscillator, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_OSCILLATOR_SAWTOOTH;


//Module - collection of systems
typedef struct MDL_OSCILLATOR
{
    SYS_OSCILLATOR_SINE         Sine;
    SYS_OSCILLATOR_SQUARE       Square;
    SYS_OSCILLATOR_TRIANGLE     Triangle;
    SYS_OSCILLATOR_SAWTOOTH     Sawtooth;
} MDL_OSCILLATOR;


typedef struct SYS_NOISE_WHITE
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_NOISE_WHITE;


typedef struct SYS_NOISE_BROWN
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (CMP_NOISE &Noise, f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_NOISE_BROWN;


typedef struct MDL_NOISE
{
    SYS_NOISE_WHITE     White;
    SYS_NOISE_BROWN     Brown;
} MDL_NOISE;


typedef struct SYS_WAV
{
    //Data
    size_t                  SystemCount;    
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_SOURCE ID);
    bool Remove             (ID_SOURCE ID);
    void RenderToBuffer     (CMP_WAV &WAV, CMP_BUFFER &Buffer, i32 Rate, size_t BufferCount);
    void Update             (ENTITY_SOURCES *Sources, f64 Pitch, size_t BufferCount);

} SYS_WAV;


typedef struct SYS_MIX
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemSources;

    void Create(MEMORY_ARENA *Arena, size_t Size);
    void Destroy(MEMORY_ARENA *Arena);
    void Add(ID_VOICE Voice);
    bool Remove(ID_VOICE ID);
    void RenderToBuffer(f32 *Channel0, f32 *Channel1, size_t SamplesToWrite, CMP_BUFFER &SourceBuffer, CMP_FADE &Amplitude, CMP_PAN &SourcePan, f64 TargetAmplitude);
    size_t Update(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, f32 *MixerChannel0, f32 *MixerChannel1, size_t SamplesToWrite);

} SYS_MIX;


typedef struct SYS_ENVELOPE_ADSR
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Size);
    void Destroy(MEMORY_ARENA *Arena);
    void Add(ID_VOICE ID);
    bool Remove(ID_VOICE ID);
    void Edit(ENTITY_VOICES *Voices, ID_VOICE ID, f64 ControlRate, f64 MaxAmplitude, f64 Attack, f64 Decay, f64 SustainAmplitude, f64 Release, bool IsAligned = true);
    void RenderToBuffer(CMP_ADSR &ADSR, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update(ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_ENVELOPE_ADSR;

typedef struct SYS_ENVELOPE_BREAKPOINT
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ARENA *Arena, size_t Size);
    void Destroy            (MEMORY_ARENA *Arena);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void CreateFromFile     (ENTITY_VOICES *Voices, ID_VOICE ID, char const *File);
    void EditPoint          (ENTITY_VOICES *Voices, ID_VOICE ID, size_t PointIndex, f64 Value, f64 Time);
    void Update             (ENTITY_VOICES *Voices, SYS_FADE *Fade, f64 Time);


} SYS_ENVELOPE_BREAKPOINT;


typedef struct SYS_PARAMETER
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Size)
    {
        SystemVoices = (ID_VOICE *) Arena->Alloc((sizeof(ID_VOICE) * Size), MEMORY_ARENA_ALIGNMENT);
        SystemCount = 0;
    }

    void Destroy(MEMORY_ARENA *Arena)
    {
        Arena->FreeAll();
    }

    void Add(ID_VOICE ID)
    {
        SystemVoices[SystemCount] = ID;
        ++SystemCount;
    }

    bool Remove(ID_VOICE ID)
    {
        for(size_t i = 0; i <= SystemCount; ++i)
        {
            if(SystemVoices[i] == ID)
            {
                SystemVoices[i] = 0;
                --SystemCount;
                return true;
            }
        }
        //!Log
        return false;
    }
    
    void Edit(ENTITY_VOICES *Voices, ID_VOICE ID, f64 Time, f64 Value, f64 Duration)
    {
        //Loop through every voice that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active voice in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //voice is valid - get component
                size_t Index = Voices->RetrieveIndex(Voice);
                // CMP_PARAMETER &Parameter = Voices->Amplitudes[Index];

                // if(Value != Parameter.CurrentValue)
	            // {
	            // 	Parameter.StartValue    = Parameter.CurrentValue;
	            // 	Parameter.EndValue      = Value;
	            // 	Parameter.StartTime     = Duration;
	            // 	Parameter.DeltaTime     = Time;
	            // 	Parameter.IsDone        = false;
	            // }
            }
        }
    }

    void Update(ENTITY_VOICES *Voices, f64 Time)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice != 0)
            {
                //Source is valid - get component
                size_t Index = Voices->RetrieveIndex(Voice);
                // CMP_PARAMETER &Parameter = Voices->Amplitudes[Index];

	            // if(Parameter.CurrentValue == Parameter.EndValue)
	            // {
                //     Parameter.IsDone = true;
                //     // Info("Done");
	            // }
	            // else
	            // {
	            // 	f64 Step = (f64) MIN((Time - Parameter.StartTime) / Parameter.DeltaTime, 1.0);
	            // 	Parameter.IsDone = Step >= 1.0f;

                //     switch(Parameter.Type)
                //     {
                //         case CMP_PARAMETER::TYPE::LINEAR:
                //         {
                //             Parameter.CurrentValue = InterpLinear(Parameter.StartValue, Parameter.EndValue, Step);
                //             printf("Linear %f\t", Parameter.CurrentValue);
                //             break;
                //         }
                //         case CMP_PARAMETER::TYPE::LOGARITHMIC:
                //         {
                //             Parameter.CurrentValue = InterpLog(Parameter.StartValue, Parameter.EndValue, Step);
                //             printf("Log %f\n", Parameter.CurrentValue);
                //             break;
                //         }            
                //     }
                //     // Linear, Square, Smooth, Fast Start, Fast End and Bezier
	            // }
            }
        }
    }


} SYS_PARAMETER;



typedef struct POLAR_MIXER
{
    size_t  Count;
    SYS_MIX **Mixes;
} POLAR_MIXER;

typedef struct MDL_SYSTEMS
{
    //Individual
    SYS_FADE                Fade;
    SYS_PARAMETER           Parameter;
    SYS_ENVELOPE_BREAKPOINT Breakpoint;
    SYS_ENVELOPE_ADSR       ADSR;
    SYS_PLAY                Play;
    SYS_WAV                 WAV;
    SYS_MIX                 Mix;  

    //Modules
    MDL_OSCILLATOR          Oscillator;
    MDL_NOISE               Noise;
} MDL_SYSTEMS;


typedef struct POLAR_POOL
{
    MEMORY_POOL Names;
    MEMORY_POOL Buffers;
    MEMORY_POOL Breakpoints;
} POLAR_POOL;


typedef struct SYS_VOICES
{
    //Data
    size_t                  SystemCount;    
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create(MEMORY_ARENA *Arena, size_t Size);
    void Destroy(MEMORY_ARENA *Arena);
    void Add(ID_SOURCE ID);
    bool Remove(ID_SOURCE ID);
    ID_VOICE Spawn(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, ID_SOURCE ID, size_t DeferCounter = 0, MDL_SYSTEMS *Systems = 0, POLAR_POOL *Pools = 0);
    void Update(ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, MDL_SYSTEMS *Systems, POLAR_POOL *Pools);

} SYS_VOICES;



typedef struct POLAR_ENGINE
{
    //Data
    f32             UpdateRate;
    f64             NoiseFloor;
    size_t          BytesPerSample;
    u32             BufferFrames;
    u32             LatencyFrames;
    CMP_FORMAT      Format;
    CMP_RINGBUFFER  CallbackBuffer;
    SYS_VOICES      VoiceSystem;
    MDL_SYSTEMS     Systems;

} POLAR_ENGINE;


//Utility code
#include "polar_dsp.cpp"
#include "polar_source.cpp"
#include "polar_OSC.cpp"


//Components
#include "component/buffer.cpp"
#include "component/ringbuffer.cpp"
#include "component/fade.cpp"
#include "component/breakpoint.cpp"
#include "component/pan.cpp"
#include "component/format.cpp"
#include "component/oscillator.cpp"
#include "component/noise.cpp"
#include "component/wav.cpp"

//Entities
#include "entity/sources.cpp"
#include "entity/voices.cpp"

//Systems
#include "system/play.cpp"
#include "system/fade.cpp"
#include "system/breakpoint.cpp"
#include "system/adsr.cpp"
#include "system/oscillator/sine.cpp"
#include "system/oscillator/square.cpp"
#include "system/oscillator/triangle.cpp"
#include "system/oscillator/sawtooth.cpp"
#include "system/noise/white.cpp"
#include "system/noise/brown.cpp"
#include "system/wav.cpp"
#include "system/voices.cpp"
#include "system/mix.cpp"
