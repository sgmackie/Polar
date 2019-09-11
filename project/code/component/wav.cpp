
void CMP_WAV::Init()
{
    Data = 0;
    BitRate = 0;
    SampleRate = 0;
    Channels = 0;
    Length = 0;
    ReadIndex = 0;
}


typedef struct WAV_HEADER 
{
    u32 ChunkID;
    u32 ChunkSize;
    u32 Format;
    u32 Subchunk1ID;
    u32 Subchunk1Size;
    u16 AudioFormat;
    u16 NumChannels;
    u32 SampleRate;
    u32 ByteRate;
    u16 BlockAlign;
    u16 BitsPerSample;
    u32 Subchunk2ID;
    u32 Subchunk2Size;
} WAV_HEADER;

void CMP_WAV::CreateFromPool(MEMORY_POOL *Pool, char const *Name, bool IsPowerOf2)
{
    // Clear to 0
    Init();

    // Open file
#if _WIN32
    FILE *File;  
    if(fopen_s(&File, Name, "rb") != 0) 
    {
        Warning("WAV: Failed to open file %s", Name);
        return;
    }   
#else
    FILE *File;
    File = fopen(Name, "rb");
    if(!File) 
    {
        Warning("WAV: Failed to open file %s", Name);
        return;
    }   
#endif

    // Read header - check chunks
    WAV_HEADER Header;
    size_t BytesReturned = fread(&Header, sizeof(WAV_HEADER), 1, File);
    assert(BytesReturned        > 0);
    assert(Header.ChunkID       == htonl(0x52494646)); // "RIFF"
    assert(Header.Format        == htonl(0x57415645)); // "WAVE"
    assert(Header.Subchunk1ID   == htonl(0x666d7420)); // "fmt "

    // Seek to "data" chunk
    while(Header.Subchunk2ID     != htonl(0x64617461)) // "data"
    {
        fseek(File, 4, SEEK_CUR);
        fread(&Header.Subchunk2ID, 4, 1, File);
    }    
    assert(Header.Subchunk2ID   == htonl(0x64617461)); // "data"
    fread(&Header.Subchunk2Size, 4, 1, File);

    // Set properties
    BitRate     = Header.BitsPerSample;
    SampleRate  = Header.SampleRate;
    Channels    = Header.NumChannels;
    Length      = Header.Subchunk2Size / (Channels * 4);

    //! fread size not working on this path
    if(BitRate == 16)
    {
        // Allocate
        i16 *Conversion     = (i16 *) malloc(sizeof(i16) * MAX_WAV_SIZE);
        size_t SamplesRead  = 0;
        SamplesRead         = fread(Conversion, sizeof(i16), Length, File);

        Data = (f32 *) malloc(sizeof(f32) * MAX_WAV_SIZE);

        for(u64 i = 0; i < SamplesRead; ++i)
        {
            Data[i] = Int16ToFloat(Conversion[i]);
        }

        free(Conversion);
    }
    else
    {
        // Round to power of 2 and fill silence
        if(IsPowerOf2)
        {
            u64 OldLength = Length;
            Length = UpperPowerOf264(Length);
            u64 Difference = Length - OldLength;

            // Allocate
            Data = (f32 *) malloc(sizeof(f32) * MAX_WAV_SIZE);
            //! Should be using pool but there's a severe memory leak - check pool block sizes
            // Data = (f32 *) Pool->Alloc();
    
            // Copy sample data
            size_t SamplesRead  = 0;
            SamplesRead         = fread(Data, sizeof(f32), OldLength, File);

            // Fill extra allocation to 0
            for(u64 i = OldLength; i < (OldLength + Difference); ++i)
            {
                Data[i] = 0.0f;
            }
        }     

        else
        {
            // Allocate
            Data = (f32 *) malloc(sizeof(f32) * MAX_WAV_SIZE);
            //! Should be using pool but there's a severe memory leak - check pool block sizes
            // Data = (f32 *) Pool->Alloc();
    
            // Copy sample data
            size_t SamplesRead  = 0;
            SamplesRead         = fread(Data, sizeof(f32), Length, File);
        }
    }



    // Close File
    fclose(File);
    File = 0;
}

void CMP_WAV::Destroy()
{
    free(Data);
}