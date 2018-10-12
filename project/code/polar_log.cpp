#include "polar_log.h"

//Print current state of render OutputThread
void polar_log_PrintObjectState(POLAR_OBJECT &Object)
{
    float32 DecibelConversion = (20 * log10(Object.StreamHandle->AmplitudeCurrent));
    
    //TODO: Print time stamp (like Wwise profiler)
    printf("Polar\t%s [ %llu || %s ]:\t%0.fHz\t%.2fdB\n", Object.ObjectName, Object.ObjectID, Object.ObjectTypeName, Object.WaveOscillator->FrequencyCurrent.load(), DecibelConversion);
}

//Print current sample rate and bit depth
void polar_log_PrintAudioFormat(AUDIO_FORMAT &audioFormat)
{
    char *EncodingText;

    switch (audioFormat.getEncoding())
    {
        case AudioEncoding::FLOATING_POINT:
        {
            EncodingText = "32-Bit Floating Point";
            break;
        }
        case AudioEncoding::PCM_8:
        {
            EncodingText = "8-Bit PCM";
            break;
        }
        case AudioEncoding::PCM_16:
        {
            EncodingText = "16-Bit PCM";
            break;
        }
        case AudioEncoding::PCM_24:
        {
            EncodingText = "24-Bit PCM";
            break;     
        }
        case AudioEncoding::PCM_24_IN_32:
        {
            EncodingText = "24-Bit PCM (32-bit Container)"; 
            break; 
        }
        case AudioEncoding::PCM_32:
        {
            EncodingText = "PCM_32 PCM";  
            break;  
        }
    }

    printf("Polar\tAudio Format:\t%dHz\t%s\n", audioFormat.getSampleRateInHz(), EncodingText);
}