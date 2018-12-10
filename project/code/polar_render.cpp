#ifndef polar_render_cpp
#define polar_render_cpp

#include "polar_render.h"

//TODO: Possible to remove CRT functions like fwrite and fseek?
i32 polar_render_StartWrite(POLAR_WAV *File, POLAR_DATA *Engine)
{
	//Assign engine values to the file header
    File->WAVHeader.AudioFormat = 3;
    File->WAVHeader.NumChannels = Engine->Channels;
    File->WAVHeader.SampleRate = Engine->SampleRate;
	File->WAVHeader.BitsPerSample = Engine->BitRate;
    File->WAVHeader.ByteRate = (File->WAVHeader.SampleRate * File->WAVHeader.NumChannels * (File->WAVHeader.BitsPerSample / 8));
	File->WAVHeader.BlockAlign = (File->WAVHeader.NumChannels * (File->WAVHeader.BitsPerSample/8));

	//Position to track when moving through the different header chunks
	size_t CurrentPosition = 0;
	u64 DataChunkSizeInitial = 0;
	u64 DataChunkDataSizeTargetWrite = DataChunkSizeInitial;

	//Write RIFF chunk and set format to WAVE file
	u32 RIFFChunkSize = 36 + (u32)DataChunkSizeInitial;
	CurrentPosition += fwrite("RIFF", 1, 4, (FILE *)File->WAVFile);	//Count = 1, Bytes = 4 (or 2)
	CurrentPosition += fwrite(&RIFFChunkSize, 1, 4, (FILE *)File->WAVFile);
	CurrentPosition += fwrite("WAVE", 1, 4, (FILE *)File->WAVFile);
	
	//Write WAV formatting chunk
	u64 FMTChunkSize = 16;
	CurrentPosition += fwrite("fmt ", 1, 4, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&FMTChunkSize, 1, 4, (FILE *)File->WAVFile);

	//Write above properties to the header
	CurrentPosition += fwrite(&File->WAVHeader.AudioFormat, 1, 2, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&File->WAVHeader.NumChannels, 1, 2, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&File->WAVHeader.SampleRate, 1, 4, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&File->WAVHeader.ByteRate, 1, 4, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&File->WAVHeader.BlockAlign, 1, 2, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&File->WAVHeader.BitsPerSample, 1, 2, (FILE *)File->WAVFile);

	//Set position in file where data chunk starts for writing samples to
	File->DataChunkDataStart = CurrentPosition;

	//Set up data chunk
	u32 DataChunkSize = (u32)DataChunkSizeInitial;
	CurrentPosition += fwrite("data", 1, 4, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&DataChunkSize, 1, 4, (FILE *)File->WAVFile);

	//Exit if chunks not written
	if(CurrentPosition != 20 + FMTChunkSize + 8)
	{
		return -1;
	}
	
	return 0;
}


POLAR_WAV *polar_render_OpenWAVWrite(const char *FilePath, POLAR_DATA *Engine)
{
	//TODO: Check where allocations are taking place (create seperate function? Move sample data allocation into this function?)
	POLAR_WAV *File = (POLAR_WAV *) VirtualAlloc(0, (sizeof *File), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

	File->DataChunkDataSize = 0;

	//MSVC specific file open function
	fopen_s(&File->WAVFile, FilePath, "wb");
	if(File->WAVFile == nullptr)
	{
		return nullptr;
	}

	if(polar_render_StartWrite(File, Engine) != 0)
	{
		fclose(File->WAVFile);
		return nullptr;
	}

	return File;
}

size_t polar_render_WriteRawWAV(POLAR_WAV *File, size_t BytesToWrite, const void *FileData)
{
	if(File == nullptr || BytesToWrite == 0 || FileData == nullptr)
	{
        return 0;
    }

	//Write the raw data into the file then increment the data chunk size
	size_t BytesWrittenToFile = fwrite(FileData, 1, BytesToWrite, (FILE *)File->WAVFile);
    File->DataChunkDataSize += BytesWrittenToFile;

    return BytesWrittenToFile;
}


u64 polar_render_WriteFloatWAV(POLAR_WAV *File, u64 SamplesToWrite, const void *FileData)
{
    if(File == nullptr || SamplesToWrite == 0 || FileData == nullptr)
	{
        return 0;
    }

	//Convert frame count to byte count
    u64 BytesToWrite = ((SamplesToWrite * File->WAVHeader.BitsPerSample) / 8);
    if(BytesToWrite > WAV_FILE_MAX_SIZE) 
	{
        return 0;
    }

	u64 BytesWrittenToFile = 0;
    const u8 *RawData = (const u8 *)FileData;
    
	//Keep writing until the next buffer of samples to write
	while(BytesToWrite > 0) 
	{
        u64 BytesToWriteThisIteration = BytesToWrite;

        if (BytesToWriteThisIteration > WAV_FILE_MAX_SIZE) 
		{
            BytesToWriteThisIteration = WAV_FILE_MAX_SIZE;
        }

		//Write the raw sample data
		size_t BytesWritten = polar_render_WriteRawWAV(File, (size_t)BytesToWriteThisIteration, RawData);
		if (BytesWritten == 0)
		{
            break;
        }

		//Decrement the byte count, increment the byte count in the current file
        BytesToWrite -= BytesWritten;
        BytesWrittenToFile += BytesWritten;
        RawData += BytesWritten;
    }

    return (BytesWrittenToFile * 8) / File->WAVHeader.BitsPerSample;
}

u32 polar_render_RIFFChunkRound(u64 RIFFChunkSize)
{
    if (RIFFChunkSize <= (0xFFFFFFFF - 36)) 
	{
        return 36 + (u32)RIFFChunkSize;
    } 
	
	else 
	{
        return 0xFFFFFFFF;
    }
}


u32 polar_render_DataChunkRound(u64 DataChunkSize)
{
    if (DataChunkSize <= 0xFFFFFFFF)
	{
        return (u32)DataChunkSize;
    } 
	else 
	{
        return 0xFFFFFFFF;
    }
}


void polar_render_CloseWAVWrite(POLAR_WAV *File)
{
	if(File == nullptr)
	{
		return;
	}

	u32 FilePadding = 0;
	FilePadding = (u32)(File->DataChunkDataSize % 2);

	//Move to RIFF chunk and write the final size
	fseek((FILE *)File->WAVFile, 4, SEEK_CUR); //Seek 4 bytes from the origin, using current position of the file pointer
	u32 RIFFChunkSize = polar_render_RIFFChunkRound(File->DataChunkDataSize);
	fwrite(&RIFFChunkSize, 1, 4, (FILE *)File->WAVFile);

	//Move to data chunk and write the final size
	fseek((FILE *)File->WAVFile, ((i32)File->DataChunkDataStart + 4), SEEK_CUR);
	u32 DataChunkSize = polar_render_DataChunkRound(File->DataChunkDataSize);                
	fwrite(&DataChunkSize, 1, 4, (FILE *)File->WAVFile);

	//Close file handle
	fclose((FILE*)File->WAVFile);

	//TODO: Again, check allocations and frees in these functions, may become seperate
	VirtualFree(File, 0, MEM_RELEASE);
}


f32 polar_render_GetPanPosition(i8 Position, f32 Amplitude, f32 PanFactor)
{
	f32 PanPosition; 

	//Left panning
	if(Position == 0)
	{
		PanPosition = Amplitude * sqrt(2.0) * (1 - PanFactor) / (2* sqrt(1 + PanFactor * PanFactor));
	}

	//Right panning
	if(Position == 1)
	{
		PanPosition = Amplitude * sqrt(2.0) * (1 + PanFactor) / (2* sqrt(1 + PanFactor * PanFactor));
	}

	return PanPosition;
}

void polar_render_FillBuffer(i8 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, BYTE *ByteBuffer, f32 *FileSamples, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue)
{
	//Cast from float pointer to BYTE pointer
	SampleBuffer = reinterpret_cast<f32 *>(ByteBuffer);

	u64 SamplesWritten = 0;

	for(i32 FrameIndex = 0; FrameIndex < FramesToWrite; ++FrameIndex)
	{
		f32 CurrentSample = Osc->Tick(Osc);
		
		for(i8 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex)
		{
			f32 PanAmp = polar_render_GetPanPosition(ChannelIndex, Amplitude, PanValue);

			//TODO: Merge these into one buffer (with memcpy for WASAPI call?)
			*SampleBuffer++ = CurrentSample * PanAmp;
			*FileSamples++ = CurrentSample * PanAmp;
		}
	}
}


void polar_UpdateRender(POLAR_DATA &Engine, POLAR_WAV *File, OSCILLATOR *Osc, f32 Amplitude, f32 PanValue)
{
	polar_WASAPI_PrepareBuffer(Engine.WASAPI, Engine.Buffer);

	polar_render_FillBuffer(Engine.Channels, Engine.Buffer.FramesAvailable, Engine.Buffer.SampleBuffer, Engine.Buffer.ByteBuffer, File->Data, Osc, Amplitude, PanValue);
	
	u64 SamplesWrittenToFile = polar_render_WriteFloatWAV(File, (Engine.Buffer.FramesAvailable * Engine.Channels), File->Data);

    char MetricsBuffer[256];
    sprintf(MetricsBuffer, "File: %llu\n", SamplesWrittenToFile);
    OutputDebugString(MetricsBuffer);

	polar_WASAPI_ReleaseBuffer(Engine.WASAPI, Engine.Buffer);
}


#endif