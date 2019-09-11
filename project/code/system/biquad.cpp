
void SYS_FILTER::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_VOICE *) Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_SYSTEM_FILTER);
    SystemCount = 0;
}

void SYS_FILTER::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_FILTER);
}

void SYS_FILTER::Add(ID_VOICE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_FILTER::Remove(ID_VOICE ID)
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

void SYS_FILTER::Edit(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency, f64 QFactor, f64 Amplitude)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Source is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BIQUAD &Biquad          = Voices->Filters[Index].Biquad[idx];
            CMP_FORMAT &Format          = Voices->Playbacks[Index].Format;

            SecondOrderParametric(&Biquad, Frequency, QFactor, Amplitude, Format.SampleRate);

			f64 s = 0;
			s++;
        }
    }
}

void SYS_FILTER::Edit2(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency, f64 Amplitude)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Source is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BIQUAD &Biquad          = Voices->Filters[Index].Biquad[idx];
            CMP_FORMAT &Format          = Voices->Playbacks[Index].Format;

            biquad_set_coef_first_order_shelf_low(&Biquad, Frequency, Amplitude, Format.SampleRate);

			f64 s = 0;
			s++;
        }
    }
}

void SYS_FILTER::Edit3(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency, f64 Amplitude)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Source is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BIQUAD &Biquad          = Voices->Filters[Index].Biquad[idx];
            CMP_FORMAT &Format          = Voices->Playbacks[Index].Format;

            biquad_set_coef_first_order_shelf_high(&Biquad, Frequency, Amplitude, Format.SampleRate);

			f64 s = 0;
			s++;
        }
    }
}

void SYS_FILTER::Edit4(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Source is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BIQUAD &Biquad          = Voices->Filters[Index].Biquad[idx];
            CMP_FORMAT &Format          = Voices->Playbacks[Index].Format;

            biquad_set_coef_first_order_lpf(&Biquad, Frequency, Format.SampleRate);

			f64 s = 0;
			s++;
        }
    }
}

void SYS_FILTER::Edit5(ENTITY_VOICES *Voices, ID_VOICE ID, size_t idx, f64 Frequency)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice == ID)
        {
            //Source is valid - get component
            size_t Index                = Voices->RetrieveIndex(Voice);
            CMP_BIQUAD &Biquad          = Voices->Filters[Index].Biquad[idx];
            CMP_FORMAT &Format          = Voices->Playbacks[Index].Format;

            biquad_set_coef_first_order_hpf(&Biquad, Frequency, Format.SampleRate);

			f64 s = 0;
			s++;
        }
    }
}


void SYS_FILTER::RenderToBuffer(TPL_FILTER *Filter, CMP_BUFFER &Buffer, size_t BufferCount)
{
	if(!Buffer.Data)
	{
        Fatal("Biquad: No buffer found!");
        return;
	}

	f64 Sample = 0;
	for(size_t i = 0; i < BufferCount; ++i)
	{
        Sample = 0;
		Sample = Filter->InputAmplitude * Buffer.Data[i];
        Sample = Process(&Filter->Biquad[0], Sample);
        Sample = Process(&Filter->Biquad[1], Sample);
        Sample = Process(&Filter->Biquad[2], Sample);
        Buffer.Data[i] = Filter->OutputAmplitude * Sample;
	}
	Sample = 0;
}

void SYS_FILTER::Update(ENTITY_VOICES *Voices, size_t BufferCount)
{
    //Loop through every source that was added to the system
    for(size_t SystemIndex = 0; SystemIndex <= SystemCount; ++SystemIndex)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[SystemIndex];
        if(Voice != 0)
        {
            //Source is valid - get component
            size_t VoiceIndex = Voices->RetrieveIndex(Voice);
            RenderToBuffer(&Voices->Filters[VoiceIndex], Voices->Playbacks[VoiceIndex].Buffer, BufferCount);
        }
    }
}
