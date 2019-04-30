
void CMP_WAV::Init(char const *Name)
{
    Data = 0;
    BitRate = 0;
    SampleRate = 0;
    Channels = 0;
    Length = 0;
    ReadIndex = 0;
    Data = drwav_open_file_and_read_pcm_frames_f32(Name, &Channels, &SampleRate, &Length);  
}