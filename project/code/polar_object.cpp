#include "polar_object.h"

POLAR_BUFFER *LoadFileNew()
{
	//
	// Lets open the file...
	//
	FILE *AudioFile = nullptr;
	fopen_s(&AudioFile,"../data/SFX16.wav","r");

	if(AudioFile)
    {
		printf("\n\nFile opened for reading");
    }

	else
	{
		printf("\n\nCouldn't open the file");
        return nullptr;
	}

	//
	// Lets read in the RIFF tag and see if we're getting this thing right
	//
	
    printf("\n\n\n\n");
    //allocate 4 bytes for the buffer
    char *RIFFData = (char *) VirtualAlloc(0, ((sizeof (char)) * 4), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    printf("\n\n\n");

    fread(RIFFData, (sizeof(char)), 4, AudioFile);

    //! Doesn't work!
	if(strcmp(RIFFData, "RIFF"))
    {
        printf("%s", RIFFData);
		printf("\n\nFound the RIFF tag ");
    }
    else
    {
        printf("\n\nNo RIFF tag!");
    }
	
	//
	// Lets get to the data tag and see if we can read it 
	//
	fseek(AudioFile, 36, SEEK_SET);
	fread(RIFFData, (sizeof(char)), 4, AudioFile);

	if(RIFFData, "data")
    {
		printf("\n\nFound the data tag");
    }
	
	//
	// OK, lets see if we cen get the data size...
	//
	i32 data_size = 0; // 32 bit value here...so use an int

	fseek(AudioFile, 40, SEEK_SET);
	fread(&data_size, (sizeof(i32)), 1, AudioFile);
	printf("\n\nThe data size of the file is %i Kb", (data_size / 1024)); 

	//
	//OK, now we know how big the data is. We now have to read the frames in a buffer.
	//A frame in bytes is number of channels * bit rate / 8. 
	//

    u32 frame_size = 2 * 16 / 8; // = 4. The magic numbers are used here because we know the file (this is just a test program)

	//
	//So we have 4 bytes per frame. This means we have data_size / frame_size number of frames...
	//
	f32 frames_in_file = (f32) data_size / (f32) frame_size;

	printf("\n\nOk so we have %.2f frames in the entire file\n\n\n\n\n", frames_in_file);

	//
	//OK, so now lets read 1 sec worth of frames into a buffer. 
	//

	//Start by allocating a buffer based around the frame
	//
    POLAR_BUFFER *FileBuffer = (POLAR_BUFFER *) VirtualAlloc(0, (sizeof FileBuffer), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    // FileBuffer->RawSamples = (BYTE *) VirtualAlloc(0, ((sizeof BYTE) * frame_size * 44100), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    FileBuffer->RawSamples = (BYTE *) VirtualAlloc(0, ((sizeof FileBuffer->RawSamples) * frame_size * 44100), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        
    
    if(FileBuffer->RawSamples)
    {
        printf("\n\nBuffer allocated");
    }

	// go to byte 44, where the data starts
	fseek(AudioFile, 44, SEEK_SET);
	fread(FileBuffer->RawSamples, sizeof(BYTE), (frame_size * 44100), AudioFile);

    return FileBuffer;
}

POLAR_OBJECT polar_object_CreateObject(u64 ID, char *Name, POLAR_OBJECT_TYPE Type)
{
    POLAR_OBJECT Result;

    Result.ObjectID = ID;
    Result.ObjectName = Name;
    Result.ObjectType = Type;

    if(Result.ObjectType == PLR_OSC)
    {
        OSCILLATOR *Osc = dsp_wave_CreateOscillator();
        Result.WaveOscillator = Osc;
        Result.ObjectTypeName = "PLR_OSC";
    }

    if(Result.ObjectType == PLR_WAV)
    {
        POLAR_BUFFER *File = LoadFileNew();
        Result.FileBuffer = File;
        Result.ObjectTypeName = "PLR_WAV";
    }

    return Result;
}

void polar_object_DestroyObject(POLAR_OBJECT Object)
{
    if(Object.ObjectType == PLR_OSC)
    {
        dsp_wave_DestroyOscillator(Object.WaveOscillator);
    }
}

POLAR_OBJECT_ARRAY polar_object_CreateObjectArray(u32 Size)
{
    POLAR_OBJECT_ARRAY Result = {};
    Result.Size = Size;
    Result.Objects = (POLAR_OBJECT **) VirtualAlloc(0, (sizeof(Result.Objects) * Result.Size), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return Result;
}

void polar_object_SubmitObject(POLAR_OBJECT_ARRAY Array, POLAR_OBJECT &Object, RENDER_STREAM *Stream, PLR_OSC_WAVEFORM Flags)
{
    Object.StreamHandle = Stream;

    if(Object.ObjectType == PLR_OSC)
    {
        switch(Flags)
        {
            case PLR_SINE:
            {
                dsp_wave_InitOscillator(Object.WaveOscillator, WAVEFORM::SINE, Object.StreamHandle->getAudioFormat()->getSampleRateInHz());
                break;
            }
            case PLR_SQUARE:
            {
                dsp_wave_InitOscillator(Object.WaveOscillator, WAVEFORM::SQUARE, Object.StreamHandle->getAudioFormat()->getSampleRateInHz());
                break;
            }
            case PLR_SAWDOWN:
            {
                dsp_wave_InitOscillator(Object.WaveOscillator, WAVEFORM::SAWDOWN, Object.StreamHandle->getAudioFormat()->getSampleRateInHz());
                break;
            }
            case PLR_SAWUP:
            {
                dsp_wave_InitOscillator(Object.WaveOscillator, WAVEFORM::SAWUP, Object.StreamHandle->getAudioFormat()->getSampleRateInHz());
                break;
            }
            case PLR_TRIANGLE:
            {
                dsp_wave_InitOscillator(Object.WaveOscillator, WAVEFORM::TRIANGLE, Object.StreamHandle->getAudioFormat()->getSampleRateInHz());
                break;
            }
            default:
            {
                debug_PrintLine(Console, "POLAR: Failed to initialise oscillator!");
            }
        }

        Stream->ObjectHandle.WaveOscillator = Object.WaveOscillator;
    }

    // if(Object.ObjectType == POLAR_OBJECT_TYPE::PLR_WAV)
    // {
    //     Stream->ObjectHandle = Object.FileBuffer;
    // }


    for(u8 i = 0; i < POLAR_MAX_OBJECTS; i++)
    {
        if(Array.Objects[i] == nullptr)
        {
            Array.Objects[i] = &Object;
        }

        else
        {
            print("Full");
        }
    }
    
    Array.Objects[0] = &Object;
}