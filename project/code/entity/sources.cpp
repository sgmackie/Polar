

void ENTITY_SOURCES::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    //Allocate space for components
    Names           = (char **)                 Allocator->Alloc((sizeof(char **) * Size), HEAP_TAG_ENTITY_SOURCE);
    IDs             = (ID_SOURCE *)             Allocator->Alloc((sizeof(ID_SOURCE) * Size), HEAP_TAG_ENTITY_SOURCE);
    Voices          = (CMP_VOICEMAP *)          Allocator->Alloc((sizeof(CMP_VOICEMAP) * Size), HEAP_TAG_ENTITY_SOURCE);
    Formats         = (CMP_FORMAT *)            Allocator->Alloc((sizeof(CMP_FORMAT) * Size), HEAP_TAG_ENTITY_SOURCE);
    Positions       = (CMP_POSITION *)          Allocator->Alloc((sizeof(CMP_POSITION) * Size), HEAP_TAG_ENTITY_SOURCE);
    Distances       = (CMP_DISTANCE *)          Allocator->Alloc((sizeof(CMP_DISTANCE) * Size), HEAP_TAG_ENTITY_SOURCE);
    Amplitudes      = (CMP_FADE *)              Allocator->Alloc((sizeof(CMP_FADE) * Size), HEAP_TAG_ENTITY_SOURCE);
    Crossfades      = (CMP_CROSSFADE *)         Allocator->Alloc((sizeof(CMP_CROSSFADE) * Size), HEAP_TAG_ENTITY_SOURCE);
    Pans            = (CMP_PAN *)               Allocator->Alloc((sizeof(CMP_PAN) * Size), HEAP_TAG_ENTITY_SOURCE);
    Types           = (TPL_TYPE *)              Allocator->Alloc((sizeof(TPL_TYPE) * Size), HEAP_TAG_ENTITY_SOURCE);
    ADSRs           = (CMP_ADSR *)              Allocator->Alloc((sizeof(CMP_ADSR) * Size), HEAP_TAG_ENTITY_SOURCE);
    Filters         = (TPL_FILTER *)            Allocator->Alloc((sizeof(TPL_FILTER) * Size), HEAP_TAG_ENTITY_SOURCE);
    Breakpoints     = (CMP_BREAKPOINT *)        Allocator->Alloc((sizeof(CMP_BREAKPOINT) * Size), HEAP_TAG_ENTITY_SOURCE);
    Modulators      = (CMP_MODULATOR *)         Allocator->Alloc((sizeof(CMP_MODULATOR) * Size), HEAP_TAG_ENTITY_SOURCE);
    Sines           = (CMP_CUDA_SINE *)         Allocator->Alloc((sizeof(CMP_CUDA_SINE) * Size), HEAP_TAG_ENTITY_SOURCE);
    Bubbles         = (TPL_BUBBLES *)           Allocator->Alloc((sizeof(TPL_BUBBLES) * Size), HEAP_TAG_ENTITY_SOURCE);
    //! This structs are over 4mb!
    // Grains          = (CMP_GRAIN *)             Allocator->Alloc((sizeof(CMP_GRAIN) * Size), HEAP_TAG_ENTITY_SOURCE);    
    // FFTs            = (CMP_FFT *)               Allocator->Alloc((sizeof(CMP_FFT) * Size), HEAP_TAG_ENTITY_SOURCE);
    Flags           = (i32 *)                   Allocator->Alloc((sizeof(i32) * Size), HEAP_TAG_ENTITY_SOURCE);

    //Initialise
    Count = 0;
}

void ENTITY_SOURCES::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_ENTITY_SOURCE);
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
    Names[Count] = (char *) Pool->Retrieve();
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

HANDLE_SOURCE ENTITY_SOURCES::AddByHash(u64 Hash)
{
    //Initialise
    Init(Count);
	
    //Assign ID
    ID_SOURCE ID	= Hash;
    IDs[Count]      = ID;
    
    //Initialise
    Flags[Count]    = 0;
    
    //Increment and return handle
    HANDLE_SOURCE Source = {ID, Count};
	++Count;
    return Source;
}


bool ENTITY_SOURCES::Remove(MEMORY_POOL *Pool, ID_SOURCE ID)
{
    for(ID_SOURCE i = 0; i < Count; ++i)
    {
        if(ID == IDs[i])
        {
            //Free and reset
            Pool->Release(Names[i]);
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

HANDLE_SOURCE ENTITY_SOURCES::RetrieveHandle(ID_SOURCE ID)
{
    HANDLE_SOURCE Source = {};
    for(size_t i = 0; i < Count; ++i)
    {
        if(ID == IDs[i])
        {
            Source.ID       = ID;
            Source.Index    = i;
        }
    }
    return Source;
}