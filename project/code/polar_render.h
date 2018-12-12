#ifndef polar_render_h
#define polar_render_h

//64 bit max size
//TODO: Check on x86 builds
#define WAV_FILE_MAX_SIZE  ((u64)0xFFFFFFFFFFFFFFFF)

//Struct to hold platform specific audio API important engine properties
typedef struct POLAR_DATA
{
	//TODO: Create union for different audio API (CoreAudio)
	WASAPI_DATA *WASAPI;
	WASAPI_BUFFER Buffer;
	WASAPI_CLOCK Clock;
	u16 Channels;
	u32 SampleRate;
	u16 BitRate;
} POLAR_DATA;

//WAV file specification "http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html"
typedef struct POLAR_WAV_HEADER
{
	u16 AudioFormat;		//1 for WAVE_FORMAT_PCM, 3 for WAVE_FORMAT_IEEE_FLOAT
	u16 NumChannels;		//2
	u32 SampleRate;			//192000
	u32 ByteRate;			//SampleRate * NumChannels * BitsPerSample/8
	u16 BlockAlign;			//NumChannels * BitsPerSample/8
	u16 BitsPerSample;		//32
	u64 DataChunkDataSize;	//Overall size of the "data" chunk
	u64 DataChunkDataStart;	//Starting byte of the data chunk
} POLAR_WAV_HEADER;

typedef struct POLAR_WAV
{
	FILE *WAVFile;
	POLAR_WAV_HEADER WAVHeader;
	f32 *Data;
	u64 TotalSampleCount;
} POLAR_WAV;

//File writing
i32 polar_render_StartWrite(POLAR_WAV *File, POLAR_DATA *Engine);
POLAR_WAV *polar_render_OpenWAVWrite(const char *FilePath, POLAR_DATA *Engine);
size_t polar_render_WriteRawWAV(POLAR_WAV *File, size_t BytesToWrite, const void *FileData);
u64 polar_render_WriteFloatWAV(POLAR_WAV *File, u64 SamplesToWrite, const void *FileData);
u32 polar_render_DataChunkRound(u64 DataChunkSize);
void polar_render_CloseWAVWrite(POLAR_WAV *File);

//Rendering
f32 polar_render_GetPanPosition(u16 Position, f32 Amplitude, f32 PanFactor);
void polar_render_FillBuffer(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, BYTE *ByteBuffer, f32 *FileSamples, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue);
void polar_UpdateRender(POLAR_DATA &Engine, POLAR_WAV *File, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue);

#endif