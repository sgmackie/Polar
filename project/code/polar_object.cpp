#include "polar_object.h"


POLAR_OBJECT polar_object_CreateObject(u64 ID, char *Name, POLAR_OBJECT_TYPE Type)
{
    POLAR_OBJECT Result;

    Result.ObjectID = ID;
    Result.ObjectName = Name;
    Result.ObjectType = Type;

    if(Result.ObjectType == POLAR_OBJECT_TYPE::PLR_OSC)
    {
        OSCILLATOR *Osc = dsp_wave_CreateOscillator();
        Result.WaveOscillator = Osc;
        Result.ObjectTypeName = "PLR_OSC";
    }

    return Result;
}

void polar_object_DestroyObject(POLAR_OBJECT Object)
{
    if(Object.ObjectType == POLAR_OBJECT_TYPE::PLR_OSC)
    {
        dsp_wave_DestroyOscillator(Object.WaveOscillator);
    }
}

void polar_object_SubmitObject(POLAR_OBJECT &Object, RENDER_STREAM *Stream, PLR_OSC_WAVEFORM Flags)
{
    Object.StreamHandle = Stream;

    if(Object.ObjectType == POLAR_OBJECT_TYPE::PLR_OSC)
    {
        if(Flags == PLR_OSC_WAVEFORM::SINE)
        {
            dsp_wave_InitOscillator(Object.WaveOscillator, WAVEFORM::SINE, Object.StreamHandle->getAudioFormat()->getSampleRateInHz());
        }

        if(Flags == PLR_OSC_WAVEFORM::SQUARE)
        {
            dsp_wave_InitOscillator(Object.WaveOscillator, WAVEFORM::SQUARE, Object.StreamHandle->getAudioFormat()->getSampleRateInHz());
        }

        Stream->ObjectHandle = Object.WaveOscillator;
    }
}