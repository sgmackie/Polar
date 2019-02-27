#ifndef polar_mixer_cpp
#define polar_mixer_cpp

void Resample(f32 *InputBuffer, u32 InputFrames, f32 *OutputBuffer, u32 OutputFrames)
{
    f32 Ratio = InputFrames / OutputFrames;
    f32 Increment = 0.0f;

    for(u32 FrameIndex = 0; FrameIndex < OutputFrames; ++FrameIndex)
    {
        i32 From = Increment;
        i32 To = From + 1;
        f32 Delta = Increment - From;

        f32 FromSample = InputBuffer[From];
        f32 ToSample = InputBuffer[To];

        OutputBuffer[FrameIndex] = Lerp(FromSample, ToSample, Delta);
        Increment += Ratio;
    }
}

POLAR_MIXER *polar_mixer_Create(MEMORY_ARENA *Arena, f64 Amplitude)
{
    Assert(Arena);

    POLAR_MIXER *Result = 0;
    Result = (POLAR_MIXER *) memory_arena_Push(Arena, Result, (sizeof (POLAR_MIXER)));
    
    if(Result)
    {
        Result->SubmixCount = 0;
        Result->FirstInList = 0;
        Result->Amplitude = DB(Amplitude);

        return Result;
    }

    printf("Polar\tERROR: Failed to create mixer\n");
    return 0;
}

void polar_mixer_Destroy(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer)
{
    if(Mixer)
    {
        memory_arena_Pull(Arena);

        Mixer->SubmixCount = 0;
        Mixer->Amplitude = 0;  
        Mixer->FirstInList = 0;  
        Mixer = 0;
    }
}

void polar_mixer_SubmixCreate(MEMORY_ARENA *Arena, POLAR_MIXER *Mixer, const char ParentUID[MAX_STRING_LENGTH], const char ChildUID[MAX_STRING_LENGTH], f64 Amplitude)
{
    u64 ChildHash = Hash(ChildUID);
    
    POLAR_SUBMIX *Result = 0;
    Result = (POLAR_SUBMIX *) memory_arena_Push(Arena, Result, (sizeof (POLAR_SUBMIX)));
    Result->UID = ChildHash;
    Result->ChildSubmixCount = 0;
    Result->Amplitude = DB(Amplitude);
    Result->FX = FX_DRY;
    Result->Containers.CurrentContainers = 0;
    Result->NextSubmix = 0;

    if(ParentUID)
    {
        u64 ParentHash = Hash(ParentUID);
        for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
        {
            if(SubmixIndex->UID == ParentHash)
            {
                if(SubmixIndex->ChildSubmix == 0)
                {
                    SubmixIndex->ChildSubmix = Result;
                    SubmixIndex->ChildSubmixCount += 1;

                    return;                
                }
            }

            for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
            {
                if(ChildSubmixIndex->UID == ParentHash)
                {
                    if(ChildSubmixIndex->ChildSubmix == 0)
                    {
                        ChildSubmixIndex->ChildSubmix = Result;
                        ChildSubmixIndex->ChildSubmixCount += 1;

                        return;
                    }
                }
            }
        }

        printf("Polar\tERROR: Failed to create child submix %s in parent submix %s\n", ChildUID, ParentUID);
    }

    if(Mixer->SubmixCount == 0)
    {
        Mixer->FirstInList = Result;
        Mixer->FirstInList->NextSubmix = 0;
        Mixer->SubmixCount += 1;

        return;
    }

    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        if(SubmixIndex->NextSubmix == 0)
        {
            SubmixIndex->NextSubmix = Result;
            Mixer->SubmixCount += 1;

            return;
        }
    }

    printf("Polar\tERROR: Failed to create submix %s\n", ChildUID);
}

void polar_mixer_SubmixDestroy(POLAR_MIXER *Mixer, const char UID[MAX_STRING_LENGTH])
{
    u64 SubmixHash = Hash(UID);

    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        POLAR_SUBMIX *PreviousSubmix = SubmixIndex;

        if(SubmixIndex->UID == SubmixHash)
        {
            if(SubmixIndex->NextSubmix == 0)
            {
                SubmixIndex = 0;
                Mixer->SubmixCount -= 1;
                
                return;   
            }
            
            POLAR_SUBMIX *NewNextSubmix = SubmixIndex->NextSubmix->NextSubmix;

            SubmixIndex = PreviousSubmix;
            SubmixIndex->NextSubmix = NewNextSubmix;
            Mixer->SubmixCount -= 1;
                
            return;
        }

        if(SubmixIndex->NextSubmix->UID == SubmixHash)
        {
            if(SubmixIndex->NextSubmix->NextSubmix == 0)
            {
                SubmixIndex = PreviousSubmix;
                SubmixIndex->NextSubmix = 0;
                Mixer->SubmixCount -= 1;
                
                return;
            }

            POLAR_SUBMIX *NewNextSubmix = SubmixIndex->NextSubmix->NextSubmix;

            SubmixIndex = PreviousSubmix;
            SubmixIndex->NextSubmix = NewNextSubmix;
            Mixer->SubmixCount -= 1;
                
            return;
        }

        for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
        {
            POLAR_SUBMIX *PreviousSubmix = ChildSubmixIndex;

            if(ChildSubmixIndex->UID == SubmixHash)
            {
                if(ChildSubmixIndex->NextSubmix == 0)
                {
                    ChildSubmixIndex = 0;
                    Mixer->SubmixCount -= 1;

                    return;   
                }

                POLAR_SUBMIX *NewNextSubmix = ChildSubmixIndex->NextSubmix->NextSubmix;

                ChildSubmixIndex = PreviousSubmix;
                ChildSubmixIndex->NextSubmix = NewNextSubmix;
                Mixer->SubmixCount -= 1;

                return;
            }

            if(ChildSubmixIndex->NextSubmix->UID == SubmixHash)
            {
                if(ChildSubmixIndex->NextSubmix->NextSubmix == 0)
                {
                    ChildSubmixIndex = PreviousSubmix;
                    ChildSubmixIndex->NextSubmix = 0;
                    Mixer->SubmixCount -= 1;

                    return;
                }

                POLAR_SUBMIX *NewNextSubmix = ChildSubmixIndex->NextSubmix->NextSubmix;

                ChildSubmixIndex = PreviousSubmix;
                ChildSubmixIndex->NextSubmix = NewNextSubmix;
                Mixer->SubmixCount -= 1;

                return;
            }
        }
    }

    printf("Polar\tERROR: Failed to destroy submix %s\n", UID);
}


void polar_mixer_ContainerCreate(POLAR_MIXER *Mixer, const char SubmixUID[MAX_STRING_LENGTH], const char ContainerUID[MAX_STRING_LENGTH], f64 Amplitude)
{
    u64 SubmixHash = Hash(SubmixUID);
    u64 ContainerHash = Hash(ContainerUID);

    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        if(SubmixIndex->UID == SubmixHash)
        {            
            for(u32 i = 0; i <= SubmixIndex->Containers.CurrentContainers; ++i)
            {
                if(SubmixIndex->Containers.UID[i] == 0)
                {
                    SubmixIndex->Containers.UID[i] = ContainerHash;
                    SubmixIndex->Containers.Amplitude[i] = DB(Amplitude);
                    SubmixIndex->Containers.FX[i] = FX_DRY;

                    SubmixIndex->Containers.CurrentContainers += 1;

                    return;
                }
            }
        }

        for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
        {
            if(ChildSubmixIndex->UID == SubmixHash)
            {
                for(u32 i = 0; i <= ChildSubmixIndex->Containers.CurrentContainers; ++i)
                {
                    if(ChildSubmixIndex->Containers.UID[i] == 0)
                    {
                        ChildSubmixIndex->Containers.UID[i] = ContainerHash;
                        ChildSubmixIndex->Containers.Amplitude[i] = DB(Amplitude);
                        ChildSubmixIndex->Containers.FX[i] = FX_DRY;

                        ChildSubmixIndex->Containers.CurrentContainers += 1;

                        return;
                    }
                }
            }
        }
    }

    printf("Polar\tERROR: Failed to create container %s in submix %s\n", ContainerUID, SubmixUID);
}

void polar_mixer_ContainerDestroy(POLAR_MIXER *Mixer, const char ContainerUID[MAX_STRING_LENGTH])
{
    u64 ContainerHash = Hash(ContainerUID);

    for(POLAR_SUBMIX *SubmixIndex = Mixer->FirstInList; SubmixIndex; SubmixIndex = SubmixIndex->NextSubmix)
    {
        for(u32 i = 0; i <= SubmixIndex->Containers.CurrentContainers; ++i)
        {
            if(SubmixIndex->Containers.UID[i] == ContainerHash)
            {
                SubmixIndex->Containers.UID[i] = 0;
                SubmixIndex->Containers.Amplitude[i] = 0;
                SubmixIndex->Containers.FX[i] = FX_DRY;

                SubmixIndex->Containers.CurrentContainers -= 1;

                return;
            }
        }

        for(POLAR_SUBMIX *ChildSubmixIndex = SubmixIndex->ChildSubmix; ChildSubmixIndex; ChildSubmixIndex = ChildSubmixIndex->ChildSubmix)
        {
            for(u32 i = 0; i <= ChildSubmixIndex->Containers.CurrentContainers; ++i)
            {
                if(ChildSubmixIndex->Containers.UID[i] == ContainerHash)
                {
                    ChildSubmixIndex->Containers.UID[i] = 0;
                    ChildSubmixIndex->Containers.Amplitude[i] = 0;
                    ChildSubmixIndex->Containers.FX[i] = FX_DRY;

                    ChildSubmixIndex->Containers.CurrentContainers -= 1;

                    return;
                }
            }
        }
    }

    printf("Polar\tERROR: Failed to destroy container %s", ContainerUID);
}

#endif