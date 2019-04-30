
void ENTITY_VOICES::Create(MEMORY_ARENA *Arena, size_t Size)
{
    //Allocate space for components
    IDs             = (ID_VOICE *)          Arena->Alloc((sizeof(ID_VOICE) * Size), MEMORY_ARENA_ALIGNMENT);
    Playbacks       = (TPL_PLAYBACK *)      Arena->Alloc((sizeof(TPL_PLAYBACK) * Size), MEMORY_ARENA_ALIGNMENT);
    States          = (CMP_STATE *)         Arena->Alloc((sizeof(CMP_STATE) * Size), MEMORY_ARENA_ALIGNMENT);
    Sources         = (HANDLE_SOURCE *)     Arena->Alloc((sizeof(HANDLE_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    Types           = (TPL_TYPE *)          Arena->Alloc((sizeof(TPL_TYPE) * Size), MEMORY_ARENA_ALIGNMENT);
    Amplitudes      = (CMP_FADE *)          Arena->Alloc((sizeof(CMP_FADE) * Size), MEMORY_ARENA_ALIGNMENT);
    Pans            = (CMP_PAN *)           Arena->Alloc((sizeof(CMP_PAN) * Size), MEMORY_ARENA_ALIGNMENT);
    Breakpoints     = (CMP_BREAKPOINT *)    Arena->Alloc((sizeof(CMP_BREAKPOINT) * Size), MEMORY_ARENA_ALIGNMENT);
    ADSRs           = (CMP_ADSR *)          Arena->Alloc((sizeof(CMP_ADSR) * Size), MEMORY_ARENA_ALIGNMENT);
    Flags           = (i32 *)               Arena->Alloc((sizeof(i32) * Size), MEMORY_ARENA_ALIGNMENT);

    //Initialise
    Count = 0;
}

void ENTITY_VOICES::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void ENTITY_VOICES::Init(size_t Index)
{
    IDs[Index]          = 0;
    States[Index].Play  = STOPPED;
    States[Index].Voice = INACTIVE;
    Flags[Index]        = 0;
}

ID_VOICE ENTITY_VOICES::Add(HANDLE_SOURCE Source)
{
    //Initialise
    Init(Count);

    //Assign ID
    u64 ID = XXH64(&Count, sizeof(Count), 0);
    IDs[Count] = ID;

    //Assign handle from source
    Sources[Count] = Source;

    //Increment
	++Count;
    return ID;
}

ID_VOICE ENTITY_VOICES::RetrieveID(size_t Index)
{
    for(size_t i = 0; i < Count; ++i)
    {
        if(i == Index)
        {
            return IDs[i];
        }
    }

	Fatal("Voices: Couldn't find entity! Index: %zu", Index);
    return -1;
}

size_t ENTITY_VOICES::RetrieveIndex(ID_VOICE ID)
{
    for(size_t i = 0; i < Count; ++i)
    {
        if(ID == IDs[i])
        {
            return i;
        }
    }

	Fatal("Voices: Couldn't find entity! ID: %llu", ID);
    return -1;
}
