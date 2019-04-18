

void ENTITY_SOURCES::Create(MEMORY_ARENA *Arena, size_t Size)
{
    //Allocate space for components
    Names           = (char **)                 Arena->Alloc((sizeof(char **) * Size), MEMORY_ARENA_ALIGNMENT);
    IDs             = (ID_SOURCE *)             Arena->Alloc((sizeof(ID_SOURCE) * Size), MEMORY_ARENA_ALIGNMENT);
    Playbacks       = (TPL_PLAYBACK *)          Arena->Alloc((sizeof(TPL_PLAYBACK) * Size), MEMORY_ARENA_ALIGNMENT);
    Positions       = (CMP_POSITION *)          Arena->Alloc((sizeof(CMP_POSITION) * Size), MEMORY_ARENA_ALIGNMENT);
    Amplitudes      = (CMP_FADE *)              Arena->Alloc((sizeof(CMP_FADE) * Size), MEMORY_ARENA_ALIGNMENT);
    Oscillators     = (CMP_OSCILLATOR *)        Arena->Alloc((sizeof(CMP_OSCILLATOR) * Size), MEMORY_ARENA_ALIGNMENT);
    WAVs            = (CMP_WAV *)               Arena->Alloc((sizeof(CMP_WAV) * Size), MEMORY_ARENA_ALIGNMENT);
    ADSRs           = (CMP_ADSR *)              Arena->Alloc((sizeof(CMP_ADSR) * Size), MEMORY_ARENA_ALIGNMENT);
    Breakpoints     = (CMP_BREAKPOINT *)        Arena->Alloc((sizeof(CMP_BREAKPOINT) * Size), MEMORY_ARENA_ALIGNMENT);
    Modulators      = (CMP_MODULATOR *)         Arena->Alloc((sizeof(CMP_MODULATOR) * Size), MEMORY_ARENA_ALIGNMENT);
    Flags           = (i32 *)                   Arena->Alloc((sizeof(i32) * Size), MEMORY_ARENA_ALIGNMENT);

    //Initialise
    Count = 0;
}

void ENTITY_SOURCES::Destroy(MEMORY_ARENA *Arena)
{
    Arena->FreeAll();
}

void ENTITY_SOURCES::Init(size_t Index)
{
    Names[Index]        = 0;
    IDs[Index]          = 0;
    Flags[Index]        = 0;
}

ID_SOURCE ENTITY_SOURCES::AddByName(MEMORY_POOL *Pool, char *Name)
{
    //Initialise
    Init(Count);

    //Assign name
    Names[Count] = (char *) Pool->Alloc();
    memcpy(Names[Count], Name, MAX_STRING_LENGTH);
	
    //Assign ID
    ID_SOURCE ID	= FastHash(Names[Count]);
    IDs[Count]      = ID;
    
    //Initialise
    Flags[Count]    = 0;
    
    //Increment and return ID
	++Count;
    return ID;
}

ID_SOURCE ENTITY_SOURCES::AddByHash(u64 Hash)
{
    //Initialise
    Init(Count);
	
    //Assign ID
    ID_SOURCE ID	= Hash;
    IDs[Count]      = ID;
    
    //Initialise
    Flags[Count]    = 0;
    
    //Increment and return ID
	++Count;
    return ID;
}


bool ENTITY_SOURCES::Remove(MEMORY_POOL *Pool, ID_SOURCE ID)
{
    for(ID_SOURCE i = 0; i < Count; ++i)
    {
        if(ID == IDs[i])
        {
            //Free and reset
            Pool->Free(Names[i]);
            Init(Count);

            //Decrement total count
            --Count;
            return true;
        }

        else
        {
            Fatal("Sources: Couldn't find entity! ID: %llu", ID);
        }
    }

    return false;
}

size_t ENTITY_SOURCES::RetrieveIndex(ID_SOURCE ID)
{
    for(size_t i = 0; i < Count; ++i)
    {
        if(ID == IDs[i])
        {
            return i;
        }
    }

	Fatal("Sources: Couldn't find entity! ID: %llu", ID);
    return -1;
}

ID_SOURCE ENTITY_SOURCES::RetrieveID(size_t Index)
{
    for(size_t i = 0; i < Count; ++i)
    {
        if(i == Index)
        {
            return IDs[i];
        }
    }

	Fatal("Sources: Couldn't find entity! Index: %zu", Index);
    return -1;
}