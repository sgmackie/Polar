
void SYS_CROSSFADE::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    SystemVoices = (ID_SOURCE *) Allocator->Alloc((sizeof(ID_SOURCE) * Size), HEAP_TAG_SYSTEM_CROSSFADE);
    SystemCount = 0;
}

void SYS_CROSSFADE::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_SYSTEM_CROSSFADE);
}

void SYS_CROSSFADE::Add(ID_SOURCE ID)
{
    SystemVoices[SystemCount] = ID;
    ++SystemCount;
}

bool SYS_CROSSFADE::Remove(ID_SOURCE ID)
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

bool SYS_CROSSFADE::Start(ENTITY_VOICES *Voices, ID_VOICE IDA, ID_VOICE IDB, f32 Duration, f32 ControlRate)
{
    //Grab the component
    for(size_t AIndex = 0; AIndex <= SystemCount; ++AIndex)
    {
        if(SystemVoices[AIndex] == IDA)
        {
            // Found voice A - get voice B
            for(size_t BIndex = 0; BIndex <= SystemCount; ++BIndex)
            {
                if(SystemVoices[BIndex] == IDB)
                {
                    CMP_CROSSFADE &ACrossfade       = Voices->Crossfades[AIndex];
                    CMP_CROSSFADE &BCrossfade       = Voices->Crossfades[BIndex];

                    // Being faded out
                    ACrossfade.IsFadingOut          = true;
                    ACrossfade.IsOver               = false;
                    ACrossfade.DurationInSamples    = (Duration * ControlRate);
                    ACrossfade.PairHandle.ID        = IDB;
                    ACrossfade.PairHandle.Index     = BIndex;

                    // Fading into
                    BCrossfade.IsFadingOut          = false;
                    BCrossfade.IsOver               = false;
                    BCrossfade.DurationInSamples    = (Duration * ControlRate);
                    BCrossfade.PairHandle.ID        = IDA;
                    BCrossfade.PairHandle.Index     = AIndex;

                    // Fade.StartValue         = Fade.Current;
                    // Fade.EndValue           = CLAMP(Amplitude, 0.0, 1.0);
                    // Fade.Duration           = MAX(Duration, 0.0);
                    // Fade.StartTime          = Time;
                    // Fade.IsFading           = true;
                    return true;
                }        
            }        
        }
    }
    //!Log
    return false;
}

void SYS_CROSSFADE::RenderToBuffer(CMP_CROSSFADE &ACrossfade, CMP_BUFFER &ABuffer, CMP_CROSSFADE &BCrossfade, CMP_BUFFER &BBuffer, size_t BufferCount)
{
    f32 Amplitude   = 0.0f;

    if(ACrossfade.IsFadingOut)
    {
        f32 A           = 1.0f;
        f32 B           = 0.0f;

        // Voice B is the primary
        switch(BCrossfade.Flag)
        {
            case CMP_CROSSFADE::LINEAR:
            {
                for(size_t i = 0; i < BufferCount; ++i)
                {
		            Amplitude      = A + BCrossfade.Iterator * (B - A) / (floor((f32) BCrossfade.DurationInSamples) - 1);
		            ++BCrossfade.Iterator;
		            ++ACrossfade.Iterator;

                    BBuffer.Data[i] = BBuffer.Data[i] * Amplitude + ABuffer.Data[i];
                    ABuffer.Data[i] = ABuffer.Data[i] * Amplitude + BBuffer.Data[i];
                }               
            }     
        }    
    }

    else
    {
        f32 A           = 0.0f;
        f32 B           = 1.0f;

        // Voice A is the primary
        switch(ACrossfade.Flag)
        {
            case CMP_CROSSFADE::LINEAR:
            {
                for(size_t i = 0; i < BufferCount; ++i)
                {
		            Amplitude      = A + ACrossfade.Iterator * (B - A) / (floor((f32) ACrossfade.DurationInSamples) - 1);
		            ++ACrossfade.Iterator;
		            ++BCrossfade.Iterator;

                    ABuffer.Data[i] = ABuffer.Data[i] * Amplitude + BBuffer.Data[i];
                    BBuffer.Data[i] = BBuffer.Data[i] * Amplitude + ABuffer.Data[i];
                }               
            }     
        }      
    }

    // switch(BCrossfade.Flag)
    // {
    //     case CMP_CROSSFADE::LINEAR:
    //     {
    //         f32 Fade = 0.0f;
    //         for(size_t i = 0; i < BufferCount; ++i)
    //         {
    //             if(ACrossfade.IsFadingOut)
    //             {
    //                 Fade            = 1.0f + ACrossfade.Iterator * (0.0f - 1.0f) / (floor((f32) ACrossfade.DurationInSamples) - 1);  
    //                 ++ACrossfade.Iterator;   
    //                 ABuffer.Data[i] = ABuffer.Data[i] * Fade + BBuffer.Data[i];                    
    //             }
    //             else
    //             {
    //                 Fade            = 1.0f + BCrossfade.Iterator * (0.0f - 1.0f) / (floor((f32) BCrossfade.DurationInSamples) - 1);  
    //                 ++BCrossfade.Iterator;   
    //                 BBuffer.Data[i] = BBuffer.Data[i] * Fade + ABuffer.Data[i];                          
    //             }
    //         }
    //     }
    //     case CMP_CROSSFADE::CONVEX:
    //     {
    //         // f32 FadeIn = 0.0f;
    //         // f32 FadeOut = 0.0f;
    //         // for(size_t i = 0; i < BufferCount; ++i)
    //         // {
    //         //     ACrossfade.A = 0.0f;
    //         //     ACrossfade.B = 1.0f;                
    //         //     FadeIn          = ACrossfade.A + ACrossfade.Iterator * (ACrossfade.B - ACrossfade.A) / (floor((f32) ACrossfade.DurationInSamples) - 1);  
    //         //     ++ACrossfade.Iterator;   

    //         //     BCrossfade.A = 1.0f;
    //         //     BCrossfade.B = 0.0f;
    //         //     FadeOut         = BCrossfade.A + BCrossfade.Iterator * (BCrossfade.B - BCrossfade.A) / (floor((f32) BCrossfade.DurationInSamples) - 1);  
    //         //     ++BCrossfade.Iterator;   

    //         //     FadeIn          = powf(FadeIn, 2);
    //         //     FadeOut         = powf(FadeOut, 2);
    //         //     // Result          = 1 - powf(Result, 2);

    //         //     BBuffer.Data[i] = BBuffer.Data[i] * FadeIn + ABuffer.Data[i];
    //         //     ABuffer.Data[i] = ABuffer.Data[i] * FadeOut + BBuffer.Data[i];
    //         // }
    //     }     
    //     // case CMP_CROSSFADE::CONVEX:
    //     // {
    //     //     BBuffer.Data[i] = BBuffer.Data[i] * pos + ABuffer.Data[i] * (1 - pos);
    //     // }              
    // } 
}

void SYS_CROSSFADE::Update(ENTITY_VOICES *Voices, size_t BufferCount)
{
    //Loop through every source that was added to the system
    for(size_t i = 0; i <= SystemCount; ++i)
    {
        //Find active sources in the system
        ID_VOICE Voice = SystemVoices[i];
        if(Voice != 0)
        {
            // Check if fade has started
            CMP_CROSSFADE &ACrossfade   = Voices->Crossfades[i];
            CMP_BUFFER &ABuffer         = Voices->Playbacks[i].Buffer;

            // Get second voice components
            CMP_CROSSFADE &BCrossfade   = Voices->Crossfades[ACrossfade.PairHandle.Index];
            CMP_BUFFER &BBuffer         = Voices->Playbacks[ACrossfade.PairHandle.Index].Buffer;

            // Render
            if(!ACrossfade.IsOver && !BCrossfade.IsOver)
            {
                RenderToBuffer(ACrossfade, ABuffer, BCrossfade, BBuffer, BufferCount);
            }

            if(ACrossfade.Iterator == ACrossfade.DurationInSamples || BCrossfade.Iterator == BCrossfade.DurationInSamples)
            {
                BCrossfade.IsOver = true;
                ACrossfade.IsOver = true;
            }               
        }
    }
}