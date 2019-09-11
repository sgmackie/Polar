//Headers
#include "co/core.h"

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

#if _WIN32
#include "../external/udp.hh"
#include "../external/oscpkt.hh"
#endif

//IMGUI
#include "../external/imgui/imgui.cpp"
#include "../external/imgui/imgui_widgets.cpp"
#include "../external/imgui/imgui_draw.cpp"
#include "../external/imgui/imgui_demo.cpp"


//Max sizes
#define MAX_SOURCES             256
#define MAX_VOICES              128
#define MAX_VOICES_PER_SOURCE   8
#define MAX_STRING_LENGTH       128
#define MAX_BUFFER_SIZE         2048
#define MAX_BUBBLE_COUNT        512
#define MAX_PARTIALS            16
#define MAX_BREAKPOINTS         64
#define MAX_WAV_SIZE            ((48000 * 600) * 2)
#define MAX_GRAINS              512 * 4
#define MAX_GRAIN_LENGTH        4800
#define MAX_GRAIN_PLAYLIST      6

// Memory tags
#define HEAP_TAG_DEBUG                  1
#define HEAP_TAG_PLATFORM               2
#define HEAP_TAG_MIXER                  3       
#define HEAP_TAG_ENTITY_SOURCE          4  
#define HEAP_TAG_ENTITY_VOICE           5  
#define HEAP_TAG_SYSTEM_VOICE           6  
#define HEAP_TAG_SYSTEM_PLAY            7  
#define HEAP_TAG_SYSTEM_POSITION        8
#define HEAP_TAG_SYSTEM_MIX             9
#define HEAP_TAG_SYSTEM_FADE            10
#define HEAP_TAG_SYSTEM_CROSSFADE       11
#define HEAP_TAG_SYSTEM_BREAKPOINT      12
#define HEAP_TAG_SYSTEM_ADSR            13
#define HEAP_TAG_SYSTEM_FILTER          14
#define HEAP_TAG_SYSTEM_OSC_SINE        15
#define HEAP_TAG_SYSTEM_OSC_SQUARE      16
#define HEAP_TAG_SYSTEM_OSC_TRIANGLE    17
#define HEAP_TAG_SYSTEM_OSC_SAWTOOTH    18
#define HEAP_TAG_SYSTEM_NSE_WHITE       19
#define HEAP_TAG_SYSTEM_NSE_BROWN       20
#define HEAP_TAG_SYSTEM_WAV             21
#define HEAP_TAG_SYSTEM_CUDA            22
#define HEAP_TAG_SYSTEM_BUBBLES         23
#define HEAP_TAG_SYSTEM_FFT             24
#define HEAP_TAG_SYSTEM_GRAIN           25

#define HEAP_TAG_UNDEFINED              64


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
    HANDLE_VOICE Handles[MAX_VOICES_PER_SOURCE];
    size_t Count;
} CMP_VOICEMAP;

typedef struct CMP_DURATION
{
    //Data    
    u64         OriginalCount;
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

    //Functions
    void Init(f32 InputX = 0, f32 InputY = 0, f32 InputZ = 0, f32 InputW = 0);

} CMP_POSITION;


typedef struct CMP_DISTANCE
{
    //Data
    f64 FromListener;
    f64 Attenuation;
    f64 MinDistance;
    f64 MaxDistance;
    f64 Rolloff;
    f64 RolloffFactor;
    bool RolloffDirty;

    //Functions
    //TODO: Scale to Unreal units
    void Init(f64 InputMinDistance = 100.0, f64 InputMaxDistance = 10000.0, f64 InputRolloff = 10.0)
    {
        FromListener = 0.0;
        Attenuation = 0.0;
        MinDistance = InputMinDistance;
        MaxDistance = InputMaxDistance;
        Rolloff = InputRolloff;
        RolloffDirty = true;
    }

} CMP_DISTANCE;


typedef struct LISTENER
{
    CMP_POSITION    Position;
    CMP_POSITION    Rotation;
} LISTENER;


typedef struct CMP_WAV
{
    //Data
    f32     *Data;
    u8      BitRate;
    u32     SampleRate;
    u32     Channels;
    u64     Length;
    u64     ReadIndex;

    //Functions
    void Init();
    void CreateFromPool(MEMORY_POOL *Pool, char const *Name, bool IsPowerOf2 = false);
    void Destroy();

} CMP_WAV;


//TODO: Create generic void * buffers to support any type
typedef struct CMP_BUFFER
{
    //Data
    size_t  Count;
    f32     *Data;

    //Functions
    void CreateFromPool(MEMORY_POOL *Pool, size_t InputCount);
    void Destroy();
    void FreeFromPool(MEMORY_POOL *Pool);
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
    void Create(MEMORY_ALLOCATOR *Allocator, size_t Type, size_t InputCount, size_t Blocks);
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


typedef struct CMP_RAMP_LINEAR
{
    f32 A;
    f32 B;
    u64 DurationInSamples;
    u64 Iterator;
    bool IsHalfway;

    void Init(f32 InputA, f32 InputB, u64 Duration)
    {
        A					= InputA;
        B					= InputB;
        DurationInSamples   = Duration;
        Iterator            = 0;
        IsHalfway           = false;
    }

    void Reset()
    {
        Iterator            = 0;
        IsHalfway           = false;
    }

    f32 Generate(bool Linerar, bool Expo, bool SCurve)
    {
        f32 Result  = 0.0f;
        if(Linerar)
        {
		    Result      = A + Iterator * (B - A) / (floor((f32) DurationInSamples) - 1);
		    ++Iterator;
        }
        if(Expo)
        {
            Result      = A + Iterator * (B - A) / (floor((f32) DurationInSamples) - 1);
            // Result      = powf(Result, 2);
            Result      = 1 - powf(Result, 2);
		    ++Iterator;
        }
        if(SCurve)
        {
            if(Iterator == (DurationInSamples / 2))
            {
                IsHalfway = true;
                Iterator = 0;
            }
                        
            f32 NEW = A + Iterator * (B - A) / (floor((f32) (DurationInSamples / 2) - 1));

            if(IsHalfway)
            {
                Result = 0.5 * (1 - powf((1 - NEW), 2)) + 0.5;        
            }
            else
            { 
                Result = 0.5 * powf(NEW, 2);
            }
            ++Iterator;
        }        

        return Result; 
    }
} CMP_RAMP_LINEAR;

typedef struct CMP_CROSSFADE
{
    //Flags
    typedef enum TYPE
    {
        LINEAR              = 1 << 0,
        CONCAVE             = 1 << 1,
        CONVEX              = 1 << 2,
        //! Not implemented
        SCURVE              = 1 << 3,
    } TYPE;

    //Data
    i32                     Flag;
    bool                    IsOver;
    bool                    IsFadingOut;
    HANDLE_VOICE            PairHandle;
    u64                     DurationInSamples;
    u64                     Iterator;

    //Functions
    void Init(f32 ControlRate, i32 Type = 0, bool InputIsFadingOut = false, f32 Duration = 1.0f)
    {
        Flag                |= Type;
        IsOver              = true;
        PairHandle.ID       = 0;
        PairHandle.Index    = 0;
        IsFadingOut         = InputIsFadingOut;
        DurationInSamples   = ControlRate * Duration;
        Iterator            = 0;
    }

} CMP_CROSSFADE;



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
    u64                     Index;
    size_t                  Count;
    CMP_BREAKPOINT_POINT    *Points;

    //Functions
    void Init               (size_t PointIndex, f64 InputValue, f64 InputTime);
    void CreateFromPool     (MEMORY_POOL *Pool, size_t InputCount);
    void FreeFromPool       (MEMORY_POOL *Pool);
    void Destroy            ();

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

//!Create filter bank of [] biquads with output gain
typedef struct CMP_BIQUAD
{
    //!TEMP
    f64 InputAmplitude;
    f64 OutputAmplitude;

    typedef enum TYPE
    {
        LOWPASS     = 1 << 0,
        HIGHPASS    = 1 << 1,
        PARAMETRIC  = 1 << 2,
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

    void Init()
    {
        f64 A0 = 0;
        f64 A1 = 0;
        f64 A2 = 0;
        f64 B1 = 0;
        f64 B2 = 0;
        f64 C0 = 0;
        f64 D0 = 0;
        f64 DelayInput1 = 0;
        f64 DelayInput2 = 0;
        f64 DelayOutput1 = 0;
        f64 DelayOutput2 = 0; 
    }

} CMP_BIQUAD;

typedef struct TPL_FILTER
{
    f64 InputAmplitude;
    f64 OutputAmplitude;    
    CMP_BIQUAD Biquad[3];
} TPL_FILTER;



typedef union TPL_TYPE
{
    CMP_OSCILLATOR  Oscillator;
    CMP_NOISE       Noise;
    CMP_WAV         WAV;
} TPL_TYPE;

typedef struct CMP_CUDA_PHASOR
{
    f32 LastPhaseIncrement;
    f32 Frequency;
    f32 Amplitude;
    f32 *PhaseIncrements;
    f64 SizeOverSampleRate;

    void Init(size_t Frames, f32 SampleRate, f32 InputFrequency, f32 InputAmplitude = 1.0f, f64 Limit = TWO_PI32)
    {
        SizeOverSampleRate = Limit / SampleRate;
        Frequency = InputFrequency;
        Amplitude = InputAmplitude;
        LastPhaseIncrement = 0;
    }
    void CreateFromPool(MEMORY_POOL *Pool)
    {
        PhaseIncrements = (f32 *) Pool->Retrieve();
    }

    void FreeFromPool(MEMORY_POOL *Pool)
    {
        Pool->Release(PhaseIncrements);
    }    

} CMP_CUDA_PHASOR;

typedef struct CMP_CUDA_SINE
{
    size_t Count;
    CMP_CUDA_PHASOR *Partials;

    void Init()
    {
        Count = 0;
        Partials = 0;
    }

    void CreateFromPool(MEMORY_POOL *Pool)
    {
        Partials = (CMP_CUDA_PHASOR *) Pool->Retrieve();
    }

    void FreeFromPool(MEMORY_POOL *Pool)
    {
        Pool->Release(Partials);
        Init();
    }

} CMP_CUDA_SINE;


typedef struct CMP_GRAIN
{
    // Data
    size_t          Count;
    size_t          Playlist[MAX_GRAIN_PLAYLIST];
    size_t          ListReader;
    CMP_RAMP_LINEAR Crossfade;
    f32             *Envelope;
    f32             *Data[MAX_GRAINS];

    // Prototypes
    void            Create();
    void            Destroy();

} CMP_GRAIN;

typedef struct CMP_FFT
{
    bool IsComputed;
    f64 *FrequencyData[MAX_GRAINS];

    void Create(u64 InputSize)
    {
        for(size_t i = 0; i < MAX_GRAINS; ++i)
        {
            FrequencyData[i] = (f64 *) malloc(sizeof(f64) * MAX_GRAIN_LENGTH);
        }
    }

    void Destroy()
    {
        for(size_t i = 0; i < MAX_GRAINS; ++i)
        {
            free(FrequencyData);
        }
    }

} CMP_FFT;


typedef struct ENTITY_SOURCES
{
    //Flags
    enum FLAG_COMPONENTS
    {
        PLAYBACK                = 1 << 0,
        AMPLITUDE               = 1 << 1,
        CROSSFADE               = 1 << 2,
        PAN                     = 1 << 3,
        POSITION                = 1 << 4,
        OSCILLATOR              = 1 << 5,
        NOISE                   = 1 << 6,
        WAV                     = 1 << 7,
        ADSR                    = 1 << 8,
        BREAKPOINT              = 1 << 9,
        MODULATOR               = 1 << 10,
        BIQUAD                  = 1 << 11,
        CUDA_SINE               = 1 << 12,
        CUDA_BUBBLE             = 1 << 13,
        GRAIN                   = 1 << 14,
        FFT                     = 1 << 14,
    };

    //Data
    size_t                      Count;
    char                        **Names;
    ID_SOURCE                   *IDs;
    CMP_VOICEMAP                *Voices;
    CMP_FORMAT                  *Formats;
    CMP_FADE                    *Amplitudes;
    CMP_CROSSFADE               *Crossfades;
    CMP_PAN                     *Pans;
    CMP_POSITION                *Positions;
    CMP_DISTANCE                *Distances;
    TPL_TYPE                    *Types;
    CMP_ADSR                    *ADSRs;
    CMP_BREAKPOINT              *Breakpoints;
    TPL_FILTER                  *Filters;
    CMP_MODULATOR               *Modulators;
    CMP_CUDA_SINE               *Sines;
    TPL_BUBBLES                 *Bubbles;
    CMP_GRAIN                   *Grains;
    CMP_FFT                     *FFTs;
    i32                         *Flags;

    //Functions
    void Create                 (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy                (MEMORY_ALLOCATOR *Allocator);
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
    STATE_PLAY  Play;
    i32         LoopCount;
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
    size_t                  Count;
    ID_VOICE                *IDs;
    TPL_PLAYBACK            *Playbacks;
    CMP_STATE               *States;
    HANDLE_SOURCE           *Sources;
    TPL_TYPE                *Types;
    CMP_FADE                *Amplitudes;
    CMP_CROSSFADE           *Crossfades;    
    CMP_PAN                 *Pans;
    CMP_BREAKPOINT          *Breakpoints;
    CMP_ADSR                *ADSRs;
    TPL_FILTER              *Filters;
    CMP_POSITION            *Positions;
    CMP_DISTANCE            *Distances;
    CMP_CUDA_SINE           *Sines;
    TPL_BUBBLES             *Bubbles;
    CMP_GRAIN               *Grains;
    CMP_FFT                 *FFTs;
    i32                     *Flags;

    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Init               (size_t Index);
    ID_VOICE Add            (HANDLE_SOURCE Source);
    bool Remove             (ID_VOICE);
    ID_VOICE RetrieveID     (size_t Index);
    size_t RetrieveIndex    (ID_VOICE ID);

} ENTITY_VOICES;



typedef struct SYS_PLAY
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    bool Start              (ENTITY_VOICES *Voices, ID_VOICE ID, f64 InputDuration, i32 LoopCount = 0, u32 Delay = 0, bool IsAligned = true);
    void Update             (ENTITY_VOICES *Voices, f64 Time, u32 PreviousSamplesWritten, u32 SamplesToWrite);

} SYS_PLAY;


typedef struct SYS_POSITION
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void Edit               (ENTITY_VOICES *Voices, ID_VOICE ID, f64 MinDistance, f64 MaxDistance, f64 Rolloff);
    void Update             (ENTITY_VOICES *Voices, LISTENER *GlobalListener, f64 NoiseFloor);

} SYS_POSITION;



typedef struct SYS_FADE
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    bool Start              (ENTITY_VOICES *Voices, ID_VOICE ID, f64 Time, f64 Amplitude, f64 Duration);
    void Update             (ENTITY_VOICES *Voices, f64 Time);

} SYS_FADE;


typedef struct SYS_CROSSFADE
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    bool Start              (ENTITY_VOICES *Voices, ID_VOICE IDA, ID_VOICE IDB, f32 Duration, f32 ControlRate);
    void RenderToBuffer     (CMP_CROSSFADE &ACrossfade, CMP_BUFFER &ABuffer, CMP_CROSSFADE &BCrossfade, CMP_BUFFER &BBuffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);
} SYS_CROSSFADE;

typedef struct SYS_OSCILLATOR_SINE
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
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
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
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
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
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
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
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
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (RANDOM_PCG *RNG, f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, RANDOM_PCG *RNG, size_t BufferCount);

} SYS_NOISE_WHITE;


typedef struct SYS_NOISE_BROWN
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (CMP_NOISE &Noise, RANDOM_PCG *RNG, f64 Amplitude, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, RANDOM_PCG *RNG, size_t BufferCount);

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
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (CMP_WAV &WAV, CMP_BUFFER &Buffer, i32 Rate, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, f64 Pitch, size_t BufferCount);

} SYS_WAV;


typedef struct SYS_MIX
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemSources;

    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE Voice);
    bool Remove             (ID_VOICE ID);
    void RenderToBuffer     (f32 *Channel0, f32 *Channel1, size_t SamplesToWrite, CMP_BUFFER &SourceBuffer, CMP_FADE &Amplitude, CMP_PAN &SourcePan, f64 TargetAmplitude);
    size_t Update           (ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, f32 *MixerChannel0, f32 *MixerChannel1, size_t SamplesToWrite);

} SYS_MIX;


typedef struct SYS_ENVELOPE_ADSR
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void Edit               (ENTITY_VOICES *Voices, ID_VOICE ID, f64 ControlRate, f64 MaxAmplitude, f64 Attack, f64 Decay, f64 SustainAmplitude, f64 Release, bool IsAligned = true);
    void RenderToBuffer     (CMP_ADSR &ADSR, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_ENVELOPE_ADSR;

typedef struct SYS_ENVELOPE_BREAKPOINT
{
    //Data
    size_t                  SystemCount;    
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void CreateFromFile     (ENTITY_VOICES *Voices, ID_VOICE ID, char const *File);
    void EditPoint          (ENTITY_VOICES *Voices, ID_VOICE ID, size_t PointIndex, f64 Value, f64 Time);
    void Update             (ENTITY_VOICES *Voices, SYS_FADE *Fade, f64 Time);


} SYS_ENVELOPE_BREAKPOINT;

typedef struct SYS_CUDA
{
    size_t SystemCount;
    ID_VOICE *SystemVoices;

    void Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
    {
        SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_CUDA);
        SystemCount = 0;
    }

    void Destroy(MEMORY_ALLOCATOR *Allocator)
    {
        Allocator->Free(0, HEAP_TAG_SYSTEM_CUDA);
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

    void AddPartial(ENTITY_VOICES *Voices, ID_VOICE ID, MEMORY_POOL *PhasePool, size_t Frames, f32 Frequency, f32 Amplitude = 1.0f)
    {
        //Loop through every voice that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active voice in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Voice is valid - get component
                size_t Index                = Voices->RetrieveIndex(Voice);
                CMP_CUDA_SINE &Sine         = Voices->Sines[Index];
                CMP_FORMAT &Format          = Voices->Playbacks[Index].Format;

                if(Sine.Count != MAX_PARTIALS)
                {
                    Sine.Partials[Sine.Count].CreateFromPool(PhasePool);
                    Sine.Partials[Sine.Count].Init(Frames, Format.SampleRate, Frequency, Amplitude);
                    ++Sine.Count;
                }
            }
        }
    }

    void Update(ENTITY_VOICES *Voices, f32 *MixingBuffer, size_t BufferCount)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice != 0)
            {
                //Source is valid - get component
                size_t VoiceIndex       = Voices->RetrieveIndex(Voice);                
                CMP_CUDA_SINE &Sine     = Voices->Sines[VoiceIndex];
                
                //Loop though every partial
                if(Sine.Count > 0)
                {
                    for(size_t j = 0; j < Sine.Count; ++j)
                    {
                        //Reset temporary mixing buffer
                        // memset(MixingBuffer, 0, (sizeof(f32) * BufferCount));

                        //Get current phasor and call CUDA kernel
                        CMP_CUDA_PHASOR &Phasor = Voices->Sines[VoiceIndex].Partials[j];
#if PARTIALS_GPU                        
                        PhasorProcess(MixingBuffer, BufferCount, Phasor.PhaseIncrements, Phasor.Frequency, Phasor.Amplitude, Phasor.SizeOverSampleRate, Phasor.LastPhaseIncrement);
#else
	                    //Calculate phase increments
	                    f32 PhaseIncrement = (Phasor.Frequency * Phasor.SizeOverSampleRate);
	                    f32 CurrentPhase = Phasor.LastPhaseIncrement + PhaseIncrement;
	                    for(size_t i = 0; i < BufferCount; ++i)
	                    {
	                    	Phasor.PhaseIncrements[i] = CurrentPhase;
	                    	CurrentPhase += PhaseIncrement;

                            //Wrap
                            while(CurrentPhase >= TWO_PI32)
                            {
                                CurrentPhase -= TWO_PI32;
                            }
                            while(CurrentPhase < 0)
                            {
                                CurrentPhase += TWO_PI32;
                            }          
                        }

                        //Kernel
	                    for(size_t i = 0; i < BufferCount; ++i)
	                    {                        
                            MixingBuffer[i] = (sin(Phasor.PhaseIncrements[i]) * Phasor.Amplitude);                        
                        }

	                    //Save last phase value
	                    Phasor.LastPhaseIncrement = Phasor.PhaseIncrements[BufferCount - 1];
#endif
                        //Mix to voice buffer
                        for(size_t i = 0; i < BufferCount; ++i)
                        {
                            Voices->Playbacks[VoiceIndex].Buffer.Data[i] += MixingBuffer[i];
                        }
                    }
                }

            }
        }
    }

} SYS_CUDA;


typedef struct SYS_BUBBLE_CPU
{
    size_t SystemCount;
    ID_VOICE *SystemVoices;

    void Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
    {
        SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_BUBBLES);
        SystemCount = 0;
    }

    void Destroy(MEMORY_ALLOCATOR *Allocator)
    {
        Allocator->Free(0, HEAP_TAG_SYSTEM_BUBBLES);
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

    void ComputeCoefficients(CMP_BUBBLES_MODEL &Model, f64 SampleRate) 
    {
        f64 R               = (f64) exp(-Model.Damping / SampleRate);
        Model.R2            = R * R;
        Model.R2CosTheta    = (f64) (2 * cos(TWO_PI32 * Model.Frequency / SampleRate) * R);
        Model.C             = (f64) sin((TWO_PI32 * Model.Frequency / SampleRate) * R);
        Model.R             = Model.C * Model.Amplitude;
    }

    void ComputeBubbles(ENTITY_VOICES *Voices, ID_VOICE ID, f64 SampleRate, size_t SamplesToWrite, RANDOM_PCG *RNG)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Source is valid - get component
                size_t Index                    = Voices->RetrieveIndex(Voice);
                TPL_BUBBLES &Bubbles            = Voices->Bubbles[Index];

                // Find radius ranges
                f64 LogMinimum                  = log(Bubbles.RadiusMinimum / 1000);
                f64 LogMaximum                  = log(Bubbles.RadiusMaximum / 1000);
                f64 LogSize                     = (LogMaximum - LogMinimum) / (Bubbles.Count - 1);

                // Reset sum
                Bubbles.LambdaSum = 0;
                f64 Curve = 0;

                // Calculate radii and lambda values
                for(size_t i = 0; i < Bubbles.Count; ++i)
                {
                    // Calculate radius from minimum size
                    Bubbles.Radii[i] = exp(LogMinimum + i * LogSize);
                    
                    // Calculate lambda values and sum together
                    Bubbles.Lambda[i] = 1.0 / pow((1000 * Bubbles.Radii[i] / Bubbles.RadiusMinimum), Bubbles.LambdaExponent);

                    // Save max value
                    if(Bubbles.Lambda[i] > Curve) 
                    {
                        Curve = Bubbles.Lambda[i];
                    }

                    // Sum
                    Bubbles.LambdaSum += Bubbles.Lambda[i];
                }

                // Divive by maxium value
                for(size_t i = 0; i < Bubbles.Count; ++i)
                {
                    Bubbles.Lambda[i] /= Curve;
                }

                for(size_t k = 0; k < Bubbles.Count; ++k)
                {
                    // Calculate frequency, amplitude and damping according to bubble radius
                    f64 Frequency                               = (f64) 3 / Bubbles.Radii[k];
                    f64 Amplitude                               = (f64) (pow(Bubbles.Radii[k] / (Bubbles.RadiusMaximum / 1000), Bubbles.AmplitudeExponent));
                    f64 Damping                                 = (f64) (Frequency * (0.043 + sqrt(Frequency) / 721));

                    // Assign and find rising frequency factor
                    Bubbles.Generators[k].Model.Frequency       = Frequency;
                    Bubbles.Generators[k].Model.FrequencyBase   = Frequency;
                    Bubbles.Generators[k].Model.Amplitude       = Amplitude;
                    Bubbles.Generators[k].Model.Damping         = Damping;
                    Bubbles.Generators[k].Model.RiseFactor      = ((SampleRate / SamplesToWrite) / (Bubbles.Generators[k].Model.RiseAmplitude * Damping));

                    // Calculate coeffiecients for the pulse generator
                    ComputeCoefficients(Bubbles.Generators[k].Model, SampleRate); 
                }        
            }
        }
    }

    void ComputeEvents(ENTITY_VOICES *Voices, ID_VOICE ID, f64 SampleRate)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Source is valid - get component
                size_t Index                    = Voices->RetrieveIndex(Voice);
                TPL_BUBBLES &Bubbles            = Voices->Bubbles[Index];
                f64 Average                     = Bubbles.LambdaSum;
                if(Average == 0)
                {
                    Average = 1;
                }

                for(size_t i = 0; i < Bubbles.Count; ++i)
                {
                    f64 Mean = Average * (Bubbles.BubblesPerSec * Bubbles.Lambda[i]);
                    Bubbles.Generators[i].Pulse.Density = Mean * Bubbles.ProbabilityExponent;
                }
            }
        }
    }

    void Edit(ENTITY_VOICES *Voices, ID_VOICE ID, f64 SampleRate, size_t SamplesToWrite, RANDOM_PCG *RNG, 
    u32 BubbleCount, f64 InputBubblesPerSec, f64 InputRadius, f64 InputAmplitude, 
    f64 InputProbability, f64 InputRiseCutoff)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Source is valid - get component
                size_t Index            = Voices->RetrieveIndex(Voice);
                TPL_BUBBLES &Bubbles    = Voices->Bubbles[Index];
                
                bool NeedsCompute = false;
                if(Bubbles.Count != BubbleCount)
                {
                    Bubbles.Count = BubbleCount;
                    NeedsCompute = true;
                }
                if(Bubbles.BubblesPerSec != InputBubblesPerSec)
                {
                    Bubbles.BubblesPerSec = InputBubblesPerSec;
                    NeedsCompute = true;
                }
                if(Bubbles.RadiusMaximum != InputRadius)
                {
                    Bubbles.RadiusMaximum = InputRadius;
                    NeedsCompute = true;
                }
                if(Bubbles.Amplitude != InputAmplitude)
                {
                    Bubbles.Amplitude = InputAmplitude;
                    NeedsCompute = true;
                }
                if(Bubbles.ProbabilityExponent != InputProbability)
                {
                    Bubbles.ProbabilityExponent = InputProbability;
                    NeedsCompute = true;
                }       
                if(Bubbles.RiseCutoff != InputRiseCutoff)
                {
                    Bubbles.RiseCutoff = InputRiseCutoff;
                    NeedsCompute = true;
                }

                if(NeedsCompute)
                {
                    ComputeEvents(Voices, Voice, SampleRate);
                    ComputeBubbles(Voices, Voice, SampleRate, SamplesToWrite, RNG);
                }
            }
        }
    }    

    f32 ComputePulseSample(CMP_BUBBLES_PULSE &Pulse, RANDOM_PCG *RNG) 
    {
        f32 Result          = 0;
        f32 Normalisation   = 4.656612873077392578125e-10; // MRG31k3p RNG

        // Density has shifted - recompute event threshold
        if(Pulse.Density != Pulse.DensityBaseline) 
        {
            Pulse.Threshold         = Pulse.Density * Pulse.OneOverControlRate;
            Pulse.Scale             = (Pulse.Threshold > 0.0 ? 2.0 / Pulse.Threshold : 0.0);
            Pulse.DensityBaseline   = Pulse.Density;
        }

        // Seed RNG
        f32 RandomValue     = 0;
        Pulse.RandomSeed    = RandomU32(RNG);
        RandomValue         = (f32) Pulse.RandomSeed * Normalisation;

        // Multiply result
        Result = Pulse.Amplitude * (RandomValue < Pulse.Threshold ? RandomValue * Pulse.Scale - 1.0 : 0.0);

        return Result;
    }

    void RenderPulse(CMP_BUBBLES_PULSE &Pulse, f64 SampleRate, RANDOM_PCG *RNG, f64 MasterAmplitude, f32 *PulseBuffer, size_t BufferCount)
    {
        for(size_t i = 0; i < BufferCount; ++i)
        {
            f32 Sample = 0.0f;
            Sample = ComputePulseSample(Pulse, RNG);
            Sample *= MasterAmplitude;
            PulseBuffer[i] = Sample;
        }
    }

    void RenderToBuffer(CMP_BUBBLES_MODEL &Model, f32 *PulseBuffer, CMP_BUFFER &Buffer, f64 RiseCutoff, f64 SampleRate, size_t BufferCount)
    {
        bool IsSilent = true;
        f32 Impulse = 0;
        f32 Threshold = 0.00000001;
        f32 TempY1 = 0.0f;
        f32 TempY2 = 0.0f;

        for(size_t i = 0; i < BufferCount; ++i)
        {
            // Check if the applied pulse is within the threshold
            if((Impulse = abs(PulseBuffer[i])) >= Threshold) 
            {
                // Start rising from the base frequency
                IsSilent = false;
                Model.RiseCounter = 0;
                Model.Frequency = Model.FrequencyBase;
                ComputeCoefficients(Model, SampleRate);

                if(Impulse > RiseCutoff) 
                {
                    Model.IsRising = true;
                } 
                else 
                {
                    Model.IsRising = false;
                }
            }
        }

        // Set frequency according to the rise factor
        if(Model.IsRising) 
        {
            Model.Frequency = (f64) (Model.FrequencyBase * (1.0 + (++Model.RiseCounter) / Model.RiseFactor));
            ComputeCoefficients(Model, SampleRate);
        }

        // Final check if silent using the last filter chache values
        if(IsSilent) 
        {
            if(abs(Model.Y1) >= Threshold || abs(Model.Y2) >= Threshold) 
            {
                IsSilent = false;
            }
        }

        if(IsSilent) 
        {     
            return;
        }

        // Mix buffer
        TempY1 = Model.Y1;
        TempY2 = Model.Y2;        
        for(size_t i = 0; i < BufferCount; ++i)
        {
            f32 Sample      = (Model.R2CosTheta * TempY1 - Model.R2 * TempY2 + Model.R * PulseBuffer[i]);
            TempY2          = TempY1;
            TempY1          = Sample;
            Buffer.Data[i]  += Sample;
        }
        Model.Y1 = TempY1;
        Model.Y2 = TempY2;         
    }

    void Update(ENTITY_VOICES *Voices, f64 SampleRate, RANDOM_PCG *RNG, f32 *PulseTemp, size_t BufferCount)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice != 0)
            {
                //Source is valid - get component
                size_t VoiceIndex       = Voices->RetrieveIndex(Voice);
                TPL_BUBBLES &Bubbles    = Voices->Bubbles[VoiceIndex];

                memset(Voices->Playbacks[VoiceIndex].Buffer.Data, 0.0f, (sizeof(f32) * BufferCount));
                for(size_t i = 0; i < Bubbles.Count; ++i)
                {        
                    RenderPulse(Bubbles.Generators[i].Pulse, SampleRate, RNG, Bubbles.Amplitude, PulseTemp, BufferCount);
				    RenderToBuffer(Bubbles.Generators[i].Model, PulseTemp, Voices->Playbacks[VoiceIndex].Buffer, Bubbles.RiseCutoff, SampleRate, BufferCount);
                }
            }
        }
    }

} SYS_BUBBLE_CPU;



typedef struct SYS_BUBBLE_GPU
{
    size_t SystemCount;
    ID_VOICE *SystemVoices;

    // GPU Data
    CUDA_BUBBLES *GPU;


    void Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
    {
        SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_BUBBLES);
        GPU = (CUDA_BUBBLES *) Allocator->Alloc(sizeof(CUDA_BUBBLES), HEAP_TAG_SYSTEM_BUBBLES);
        SystemCount = 0;

        // Device memory
        cuda_BubblesCreate(GPU);
    }

    void Destroy(MEMORY_ALLOCATOR *Allocator)
    {
        Allocator->Free(0, HEAP_TAG_SYSTEM_BUBBLES);
        cuda_BubblesDestroy(GPU);
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

    void ComputeBubbles(ENTITY_VOICES *Voices, ID_VOICE ID, f64 SampleRate, size_t SamplesToWrite, RANDOM_PCG *RNG)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Source is valid - get component
                size_t Index                    = Voices->RetrieveIndex(Voice);

                BEGIN_BLOCK("Cuda - Bubbles");
                cuda_BubblesComputeModel(GPU, &Voices->Bubbles[Index], SampleRate, SamplesToWrite);    
                END_BLOCK();      
            }
        }
    }

    void ComputeEvents(ENTITY_VOICES *Voices, ID_VOICE ID, f64 SampleRate)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Source is valid - get component
                size_t Index = Voices->RetrieveIndex(Voice);

                BEGIN_BLOCK("Cuda - Events");
                cuda_BubblesComputeEvents(GPU, &Voices->Bubbles[Index]);
                END_BLOCK();                
            }
        }
    }

    void Edit(ENTITY_VOICES *Voices, ID_VOICE ID, f64 SampleRate, size_t SamplesToWrite, RANDOM_PCG *RNG, 
    u32 BubbleCount, f64 InputBubblesPerSec, f64 InputRadius, f64 InputAmplitude, 
    f64 InputProbability, f64 InputRiseCutoff)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Source is valid - get component
                size_t Index            = Voices->RetrieveIndex(Voice);
                TPL_BUBBLES &Bubbles    = Voices->Bubbles[Index];
                
                bool NeedsCompute = false;
                if(Bubbles.Count != BubbleCount)
                {
                    Bubbles.Count = BubbleCount;
                    NeedsCompute = true;
                }
                if(Bubbles.BubblesPerSec != InputBubblesPerSec)
                {
                    Bubbles.BubblesPerSec = InputBubblesPerSec;
                    NeedsCompute = true;
                }
                if(Bubbles.RadiusMaximum != InputRadius)
                {
                    Bubbles.RadiusMaximum = InputRadius;
                    NeedsCompute = true;
                }
                if(Bubbles.Amplitude != InputAmplitude)
                {
                    Bubbles.Amplitude = InputAmplitude;
                    NeedsCompute = true;
                }
                if(Bubbles.ProbabilityExponent != InputProbability)
                {
                    Bubbles.ProbabilityExponent = InputProbability;
                    NeedsCompute = true;
                }       
                if(Bubbles.RiseCutoff != InputRiseCutoff)
                {
                    Bubbles.RiseCutoff = InputRiseCutoff;
                    NeedsCompute = true;
                }

                if(NeedsCompute)
                {
                    ComputeEvents(Voices, Voice, SampleRate);
                    ComputeBubbles(Voices, Voice, SampleRate, SamplesToWrite, RNG);
                }
            }
        }
    }    

    void Update(ENTITY_VOICES *Voices, f64 SampleRate, RANDOM_PCG *RNG, f32 *PulseTemp, size_t BufferCount)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice != 0)
            {
                //Source is valid - get component
                size_t VoiceIndex       = Voices->RetrieveIndex(Voice);
                memset(Voices->Playbacks[VoiceIndex].Buffer.Data, 0.0f, (sizeof(f32) * BufferCount));

                BEGIN_BLOCK("Cuda - Update");
                for(size_t i = 0; i < BufferCount; ++i)
                {
                    PulseTemp[i] = RandomF32(RNG);
                }                
                cuda_BubblesUpdate(GPU, &Voices->Bubbles[VoiceIndex], SampleRate, RandomU32(RNG), PulseTemp, Voices->Playbacks[VoiceIndex].Buffer.Data, BufferCount);
                END_BLOCK();              
           
            }
        }
    }

} SYS_BUBBLE_GPU;


typedef struct SYS_GRAIN
{
    // Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;

    // Prototypes
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void ComputeEnvelope    (f32 *Envelope, f32 Attack, f32 Release, i32 ControlRate, size_t Length);
    void Compute            (ENTITY_VOICES *Voices, ID_VOICE ID, i32 SampleRate, RANDOM_PCG *RNG, f32 Density, f32 LengthInMS, f32 Delay, f32 Attack, f32 Release, size_t CrossfadeDuration);
    void ArrayShuffle       (size_t *Array, size_t Count);
    void RenderToBuffer     (CMP_GRAIN &Grain, CMP_BUFFER &Buffer, size_t BufferCount, size_t GrainSelector);
    void Update             (ENTITY_VOICES *Voices, size_t BufferCount, size_t GrainSelector);

} SYS_GRAIN;



typedef struct SYS_FFT
{
    size_t SystemCount;
    ID_VOICE *SystemVoices;

    void Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
    {
        SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_FFT);
        SystemCount = 0;
    }

    void Destroy(MEMORY_ALLOCATOR *Allocator)
    {
        Allocator->Free(0, HEAP_TAG_SYSTEM_FFT);
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

    void SwapF32(f32 &A, f32 &B)
    {
        f32 Temp = A;
        A = B;
        B = Temp;
    }

    void Compute(ENTITY_VOICES *Voices, ID_VOICE ID, i32 SampleRate)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice == ID)
            {
                //Source is valid - get component
                size_t Index                    = Voices->RetrieveIndex(Voice);
                CMP_FFT &FFTData                = Voices->FFTs[Index];
                CMP_WAV &WAV                    = Voices->Types[Index].WAV;

                //new complex array of size n=2*sample_rate
                f32 *vector = (f32 *) malloc(sizeof(f32) * (8 * SampleRate));

                u64 Offset = 0;
                u64 ChunkLength = (WAV.Length / MAX_GRAINS);

                f64 Domi[MAX_GRAINS];

                // Create an FFT array where the index = the frequency and the array value = it's gain
                for(size_t k = 0; k < MAX_GRAINS; ++k)
                {
                    f64 BaseFrequency = 67.6;
				    size_t EngineCylinders = 2;
                    f64 GrainLength = (((1 / BaseFrequency) * EngineCylinders) * SampleRate);

                    memset(vector, 0, sizeof(f32) * (8 * SampleRate));
                    
                    //variables for the fft
                    i32 n, mmax, m, j, istep, i;
                    f64 wtemp, wr, wpr, wpi, wi, theta, tempr, tempi;

                    //put the real array in a complex array
                    for(n = 0; n < SampleRate; n++)
                    {
                        if(n < WAV.Length)
                        {              
                            vector[2 * n] = (WAV.Data[n + Offset] / 32768.0f) * 8.0f;
                        }
                        else
                        {
                            vector[2 * n] = 0;
                        }   

                        vector[2 * n + 1] = 0;
                    }     

                    n = SampleRate * 2;
                    j = 0;
                    for(i = 0; i < n / 2; i += 2)
                    {
                        if(j > i)
                        {
                            SwapF32(vector[j],      vector[i]);
                            SwapF32(vector[j + 1],  vector[i + 1]);

                            if((j / 2) < (n / 4))
                            {
                                SwapF32(vector[(n - (i + 2))],      vector[(n - (j + 2))]);
                                SwapF32(vector[(n - (i + 2)) + 1],  vector[(n - (j + 2)) + 1]);
                            }                        
                        }

                        m = n >> 1;

                        while(m >= 2 && j >= m)
                        {
                            j -= m;
                            m >>= 1;
                        }

                        j += m;                    
                    }

                    //Danielson-Lanzcos routine
                    mmax = 2;
                    while(n > mmax)
                    {
                        istep = mmax << 1;
                        theta = (2 * PI32) / mmax;
                        wtemp = sin(0.5 * theta);
                        wpi = sin(theta);
                        wpr = -2.0 * wtemp * wtemp;

                        wr = 1.0;
                        wi = 0.0;

                        for(m = 1; m < mmax; m += 2)
                        {
                            for(i = m; i <= n; i += istep)
                            {
                                j               = i + mmax;
                                tempr           = wr * vector[j - 1] - wi * vector[j];
                                tempi           = wr * vector[j] + wi * vector[j - 1];
                                vector[j - 1]   = (f32) (vector[i - 1] - tempr);
                                vector[j]       = (f32) (vector[i] - tempi);
                                vector[i - 1]   += (f32) tempr;
                                vector[i]       += (f32) tempi;
                            }    

                            wr = (wtemp = wr) * wpr - wi * wpi + wr;
                            wi = wi * wpr + wtemp * wpi + wi;                
                        }

                        mmax = istep;
                    }

                    //calculate the volume values for the frequencies from the complex array
				    size_t Resize = (SampleRate / 2) + 1;
                    FFTData.FrequencyData[k] = (f64 *) realloc(FFTData.FrequencyData[k], sizeof(f64) * Resize);
                    
                    
                    for(i = 2; i <= SampleRate; i += 2)
                    {
                        f64 value = sqrt((pow(vector[i], 2) + pow(vector[i + 1], 2)));
                        //value *= (WAV.Length / SampleRate);
                        FFTData.FrequencyData[k][i / 2] = value;
                    }                
                    
                    f64 LargestValue = FFTData.FrequencyData[k][0];
                    for(i = 0; i < Resize; ++i)
                    {
		                if(LargestValue < FFTData.FrequencyData[k][i])
                        {
                            LargestValue = FFTData.FrequencyData[k][i];                        
                        }
                    }

                    Domi[k] = LargestValue;

                    Offset += ChunkLength;

                    size_t arrayidx = 475 * (WAV.Length / SampleRate);
                    f64 Value1 = FFTData.FrequencyData[k][arrayidx];
                }   

                // Free
                free(vector);                
            }
        }
    }    

    void RenderToBuffer(CMP_FFT &FFT, CMP_BUFFER &Buffer, size_t BufferCount)
    {
        for(size_t i = 0; i < BufferCount; ++i)
        {
            f32 Sample      = 0;
            Sample          = FFT.FrequencyData[0][i];
            Buffer.Data[i]  = Sample;
        }        
    }

    void Update(ENTITY_VOICES *Voices, size_t BufferCount)
    {
        //Loop through every source that was added to the system
        for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
        {
            //Find active sources in the system
            ID_VOICE Voice = SystemVoices[SystemIndex];
            if(Voice != 0)
            {
                //Source is valid - get component
                size_t VoiceIndex       = Voices->RetrieveIndex(Voice);
                RenderToBuffer(Voices->FFTs[VoiceIndex], Voices->Playbacks[VoiceIndex].Buffer, BufferCount);

            }
        }
    }

} SYS_FFT;





typedef struct SYS_FILTER
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_VOICE ID);
    bool Remove             (ID_VOICE ID);
    void biquad_set_coef_first_order_shelf_low(CMP_BIQUAD *bq, f64 freq, f64 gain, int samplerate) 
    {
        f64 u = pow(10.0, gain / 20.0);
        f64 w = (2.0 * (f64) PI32 * freq) / (f64) samplerate;
        f64 v = 4.0 / (1.0 + u);
        f64 x = v * tan(w / 2.0);
        f64 y = (1.0 - x) / (1.0 + x);
        
        bq->A0 = (1.0 - y) / 2.0;
        bq->A1 = bq->A0;
        bq->A2 = 0.0;
        bq->B1 = -y;
        bq->B2 = 0.0;
        bq->C0 = u - 1.0;
        bq->D0 = 1.0;
    }
    void biquad_set_coef_first_order_shelf_high(CMP_BIQUAD *bq, f64 freq, f64 gain, int samplerate) 
    {
        f64 u = pow(10.0, gain / 20.0);
        f64 w = (2.0 * (f64) PI32 * freq) / (f64) samplerate;
        f64 v = (1.0 + u) / 4.0;
        f64 x = v * tan(w / 2.0);
        f64 y = (1.0 - x) / (1.0 + x);

        bq->A0 = (1.0 + y) / 2.0;
        bq->A1 = -bq->A0;
        bq->A2 = 0.0;
        bq->B1 = -y;
        bq->B2 = 0.0;
        bq->C0 = u - 1.0;
        bq->D0 = 1.0;
    }    
    void SecondOrderParametric(CMP_BIQUAD *bq, f64 freq, f64 q, f64 gain, int samplerate)
    {
        f64 u = pow(10.0, gain / 20.0);
        f64 v = 4.0 / (1.0 + u);
        f64 w = (2.0 * (f64) PI32 * freq) / (f64) samplerate;
        f64 x = tan(w / (2.0 * q));
        f64 vx = v * x;
        f64 y = 0.5 * ((1.0 - vx) / (1.0 + vx));
        f64 z = (0.5 + y) * cos(w);

        //Set coeffecients
        bq->A0 = 0.5 - y;
        bq->A1 = 0.0;
        bq->A2 = -bq->A0;
        bq->B1 = -2.0 * z;
        bq->B2 = 2.0 * y;
        bq->C0 = u - 1.0;
        bq->D0 = 1.0;
    }
    void biquad_set_coef_first_order_lpf(CMP_BIQUAD *bq, f64 freq, int samplerate)
    {
        f64 x = (2.0 * (f64) PI32 * freq) / (f64) samplerate;
        f64 y = cos(x) / (1.0 + sin(x));
        
        bq->A0 = (1.0 - y) / 2.0;
        bq->A1 = bq->A0;
        bq->A2 = 0.0;
        bq->B1 = -y;
        bq->B2 = 0.0;
        bq->C0 = 1.0;
        bq->D0 = 0.0;
    }

    void biquad_set_coef_first_order_hpf(CMP_BIQUAD *bq, f64 freq, int samplerate)
    {
        f64 x = (2.0 * (f64) PI32 * freq) / (f64) samplerate;
        f64 y = cos(x) / (1.0 + sin(x));
        
        bq->A0 = (1.0 + y) / 2.0;
        bq->A1 = -bq->A0;
        bq->A2 = 0.0;
        bq->B1 = -y;
        bq->B2 = 0.0;
        bq->C0 = 1.0;
        bq->D0 = 0.0;
    }

    f64 Process(CMP_BIQUAD *bq, f64 sample)
    {

        f64 y = (bq->A0 * sample) + (bq->A1 * bq->DelayInput1) + (bq->A1 * bq->DelayInput2);
        bq->DelayInput2 = bq->DelayInput1;
        bq->DelayInput1 = sample;
        bq->DelayOutput2 = bq->DelayOutput1;
        bq->DelayOutput1 = y;
        f64 Result = (y * bq->C0) + (sample * bq->D0);

        return Result;
    }
    void Edit(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency, f64 QFactor, f64 Amplitude);
    void Edit2(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency, f64 Amplitude);
    void Edit3(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency, f64 Amplitude);
    void Edit4(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency);
    void Edit5(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency);
    void RenderToBuffer(TPL_FILTER *Filter, CMP_BUFFER &Buffer, size_t BufferCount);
    void Update(ENTITY_VOICES *Voices, size_t BufferCount);

} SYS_FILTER;






typedef struct SYS_PARAMETER
{    
    //Data
    size_t                  SystemCount;
    ID_VOICE                *SystemVoices;
    
    //Functions
    void Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
    {
        SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_UNDEFINED);
        SystemCount = 0;
    }

    void Destroy(MEMORY_ALLOCATOR *Allocator)
    {
        Allocator->Free(0, HEAP_TAG_UNDEFINED);
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
    SYS_POSITION            Position;
    SYS_FADE                Fade;
    SYS_CROSSFADE           Crossfade;
    SYS_PARAMETER           Parameter;
    SYS_ENVELOPE_BREAKPOINT Breakpoint;
    SYS_ENVELOPE_ADSR       ADSR;
    SYS_FILTER              Filter;
    SYS_PLAY                Play;
    SYS_WAV                 WAV;
    SYS_MIX                 Mix;
    SYS_CUDA                Cuda;
#if BUBBLES_GPU    
    SYS_BUBBLE_GPU          Bubbles;
#else
    SYS_BUBBLE_CPU          Bubbles;
#endif
    SYS_GRAIN               Grain;
    SYS_FFT                 FFT;

    //Modules
    MDL_OSCILLATOR          Oscillator;
    MDL_NOISE               Noise;
} MDL_SYSTEMS;


typedef struct POLAR_BUBBLE_POOL
{
    MEMORY_POOL Generators;
    MEMORY_POOL Radii;
    MEMORY_POOL Lambda;
} POLAR_BUBBLE_POOL;

typedef struct POLAR_POOL
{
    MEMORY_POOL Names;
    MEMORY_POOL Buffers;
    MEMORY_POOL Partials;
    MEMORY_POOL Phases;
    MEMORY_POOL Breakpoints;
    MEMORY_POOL WAVs;

    POLAR_BUBBLE_POOL Bubble;

} POLAR_POOL;



typedef struct SYS_VOICES
{
    //Data
    size_t                  SystemCount;    
    ID_SOURCE               *SystemSources;
    
    //Functions
    void Create             (MEMORY_ALLOCATOR *Allocator, size_t Size);
    void Destroy            (MEMORY_ALLOCATOR *Allocator);
    void Add                (ID_SOURCE ID);
    bool Remove             (ID_SOURCE ID);
    ID_VOICE Spawn          (ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, ID_SOURCE ID, f32 SampleRate = 0, u32 Random = 0, size_t DeferCounter = 0, MDL_SYSTEMS *Systems = 0, POLAR_POOL *Pools = 0);
    void Update             (ENTITY_SOURCES *Sources, ENTITY_VOICES *Voices, MDL_SYSTEMS *Systems, POLAR_POOL *Pools);

} SYS_VOICES;



typedef struct POLAR_ENGINE
{
    //Data
    f32             UpdateRate;
    f64             NoiseFloor;
    size_t          BytesPerSample;
    u32             BufferFrames;
    u32             LatencyFrames;
    RANDOM_PCG       RNG;
    CMP_FORMAT      Format;
    CMP_RINGBUFFER  CallbackBuffer;
    SYS_VOICES      VoiceSystem;
    MDL_SYSTEMS     Systems;

    void Init(u32 Seed, f32 InputUpdateRate)
    {
        UpdateRate          = 0;
        NoiseFloor          = 0;
        BytesPerSample      = 0;
        BufferFrames        = 0;
        LatencyFrames       = 0;
        RNG                 = {};
        Format              = {};
        CallbackBuffer      = {};
        VoiceSystem         = {};
        Systems             = {};    

        // PCG seed
        RandomSeed(&RNG, Seed);

        // Rate
        UpdateRate          = InputUpdateRate;
    }

} POLAR_ENGINE;



//Utility code
#include "polar_dsp.cpp"
#include "polar_source.cpp"

#if _WIN32
#include "polar_OSC.cpp"
#endif

//Components
#include "component/buffer.cpp"
#include "component/ringbuffer.cpp"
#include "component/position.cpp"
#include "component/fade.cpp"
#include "component/breakpoint.cpp"
#include "component/pan.cpp"
#include "component/format.cpp"
#include "component/oscillator.cpp"
#include "component/noise.cpp"
#include "component/wav.cpp"
#include "component/grain.cpp"
#include "component/bubble.cpp"

//Entities
#include "entity/sources.cpp"
#include "entity/voices.cpp"

//Systems
#include "system/play.cpp"
#include "system/position.cpp"
#include "system/fade.cpp"
#include "system/crossfade.cpp"
#include "system/breakpoint.cpp"
#include "system/adsr.cpp"
#include "system/oscillator/sine.cpp"
#include "system/oscillator/square.cpp"
#include "system/oscillator/triangle.cpp"
#include "system/oscillator/sawtooth.cpp"
#include "system/noise/white.cpp"
#include "system/noise/brown.cpp"
#include "system/wav.cpp"
#include "system/grain.cpp"
#include "system/biquad.cpp"
#include "system/voices.cpp"
#include "system/mix.cpp"
