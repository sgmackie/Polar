#ifndef polar_render_cpp
#define polar_render_cpp

//TODO: Possible to remove CRT functions like fwrite and fseek?
//TODO: Add BWAV support "https://tech.ebu.ch/docs/tech/tech3285.pdf"
bool polar_render_WAVWriteHeader(POLAR_WAV *File, POLAR_DATA *Engine)
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
	File->WAVHeader.DataChunkDataStart = CurrentPosition;

	//Set up data chunk
	u32 DataChunkSize = (u32)DataChunkSizeInitial;
	CurrentPosition += fwrite("data", 1, 4, (FILE *)File->WAVFile);
	CurrentPosition += fwrite(&DataChunkSize, 1, 4, (FILE *)File->WAVFile);

	//Exit if chunks not written
	if(CurrentPosition != 20 + FMTChunkSize + 8)
	{
		return false;
	}
	
	return true;
}


POLAR_WAV *polar_render_WAVWriteCreate(const char *FilePath, POLAR_DATA *Engine)
{
	//Allocate memory
#if WIN32
	POLAR_WAV *File = (POLAR_WAV *) VirtualAlloc(0, (sizeof *File), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
	File->Data = (f32 *) VirtualAlloc(0, ((sizeof *File->Data) * ((Engine->BufferFrames * Engine->Channels))), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
	POLAR_WAV *File = (POLAR_WAV *) malloc((sizeof *File));
	File->Data = (f32 *) malloc(((sizeof *File->Data) * ((Engine->BufferFrames * Engine->Channels))));
#endif
	File->Path = FilePath;

	//Open file handle
	fopen_s(&File->WAVFile, File->Path, "wb"); //MSVC specific file open
	if(File->WAVFile == nullptr)
	{
		return nullptr;
	}

	//Create WAV header
	File->WAVHeader.DataChunkDataSize = 0;
	if(polar_render_WAVWriteHeader(File, Engine) == false)
	{
		fclose(File->WAVFile);
		return nullptr;
	}

	return File;
}

size_t polar_render_WAVWriteRaw(POLAR_WAV *File, size_t BytesToWrite, const void *FileData)
{
	if(File == nullptr || BytesToWrite == 0 || FileData == nullptr)
	{
	    return 0;
	}

	//Write the raw data into the file then increment the data chunk size
	size_t BytesWrittenToFile = fwrite(FileData, 1, BytesToWrite, (FILE *)File->WAVFile);
	File->WAVHeader.DataChunkDataSize += BytesWrittenToFile;

    return BytesWrittenToFile;
}


u64 polar_render_WAVWriteFloat(POLAR_WAV *File, u64 SamplesToWrite, const void *FileData)
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

		if(BytesToWriteThisIteration > WAV_FILE_MAX_SIZE) 
		{
			BytesToWriteThisIteration = WAV_FILE_MAX_SIZE;
		}

		//Write the raw sample data
		size_t BytesWritten = polar_render_WAVWriteRaw(File, (size_t)BytesToWriteThisIteration, RawData);
		
		if(BytesWritten == 0)
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
	if(RIFFChunkSize <= (0xFFFFFFFF - 36)) 
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
	if(DataChunkSize <= 0xFFFFFFFF)
	{
		return (u32)DataChunkSize;
	}

	else 
	{
		return 0xFFFFFFFF;
	}
}


void polar_render_WAVWriteDestroy(POLAR_WAV *File)
{
	if(File == nullptr)
	{
		return;
	}

	u32 FilePadding = 0;
	FilePadding = (u32)(File->WAVHeader.DataChunkDataSize % 2);

	//Move to RIFF chunk and write the final size
	fseek((FILE *)File->WAVFile, 4, SEEK_CUR); //Seek 4 bytes from the origin, using current position of the file pointer
	u32 RIFFChunkSize = polar_render_RIFFChunkRound(File->WAVHeader.DataChunkDataSize);
	fwrite(&RIFFChunkSize, 1, 4, (FILE *)File->WAVFile);

	//Move to data chunk and write the final size
	fseek((FILE *)File->WAVFile, ((i32)File->WAVHeader.DataChunkDataStart + 4), SEEK_CUR);
	u32 DataChunkSize = polar_render_DataChunkRound(File->WAVHeader.DataChunkDataSize);                
	fwrite(&DataChunkSize, 1, 4, (FILE *)File->WAVFile);

	//Close file handle
	fclose((FILE*)File->WAVFile);

#if WIN32
	VirtualFree(File->Data, 0, MEM_RELEASE);
	VirtualFree(File, 0, MEM_RELEASE);
#else
	free(File->Data);
	free(File);
#endif
}


f32 polar_render_PanPositionGet(u16 Position, f32 Amplitude, f32 PanFactor)
{
	f32 PanPosition = Amplitude; 

	//Left panning
	if(Position % 2 == 0)
	{
		PanPosition = Amplitude * (f32) sqrt(2.0) * (1 - PanFactor) / (2* (f32) sqrt(1 + PanFactor * PanFactor));
	}

	//Right panning
	if(Position % 2 != 0)
	{
		PanPosition = Amplitude * (f32) sqrt(2.0) * (1 + PanFactor) / (2* (f32) sqrt(1 + PanFactor * PanFactor));
	}

	return PanPosition;
}

void polar_render_BufferFill(u16 ChannelCount, u32 FramesToWrite, f32 *SampleBuffer, BYTE *ByteBuffer, f32 *FileSamples, OSCILLATOR *Osc, f32 Amplitude, f32 Pan)
{
	//Cast from float to BYTE
	SampleBuffer = reinterpret_cast<f32 *>(ByteBuffer);
	
	f32 CurrentSample = 0;
	f32 PanAmp = 0;

	//Increase frame counter by the number of channels
	for(u32 FrameIndex = 0; FrameIndex < FramesToWrite; FrameIndex += ChannelCount)
	{
		//TODO: Taylor series GPU for sine call, matrix additive function
		CurrentSample = (f32) Osc->Tick(Osc);

		for(u16 ChannelIndex = 0; ChannelIndex < ChannelCount; ++ChannelIndex)
		{
			PanAmp = polar_render_PanPositionGet(ChannelIndex, Amplitude, Pan);

			SampleBuffer[FrameIndex + ChannelIndex] = CurrentSample * PanAmp;

			if(FileSamples != nullptr)
			{
				FileSamples[FrameIndex + ChannelIndex] = CurrentSample * PanAmp;
			}
		}		
	}

	memcpy(ByteBuffer, SampleBuffer, sizeof(SampleBuffer));
}


void polar_render_BufferCopy(POLAR_DATA &Engine, POLAR_WAV *File, OSCILLATOR *Osc, f32 Amplitude, f32 Pan)
{
	polar_WASAPI_BufferGet(Engine.WASAPI, Engine.Buffer);

	//TODO: How expensive are if else statements in the rendering loop?
	if(File != nullptr)
	{
		polar_render_BufferFill(Engine.Channels, (Engine.Buffer.FramesAvailable * Engine.Channels), Engine.Buffer.SampleBuffer, Engine.Buffer.ByteBuffer, File->Data, Osc, Amplitude, Pan);
		File->TotalSampleCount += polar_render_WAVWriteFloat(File, (Engine.Buffer.FramesAvailable * Engine.Channels), File->Data);
	}
	else
	{
		polar_render_BufferFill(Engine.Channels, (Engine.Buffer.FramesAvailable * Engine.Channels), Engine.Buffer.SampleBuffer, Engine.Buffer.ByteBuffer, nullptr, Osc, Amplitude, Pan);
	}

	polar_WASAPI_BufferRelease(Engine.WASAPI, Engine.Buffer);
}

#endif