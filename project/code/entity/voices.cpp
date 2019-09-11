
void ENTITY_VOICES::Create(MEMORY_ALLOCATOR *Allocator, size_t Size)
{
    //Allocate space for components
    IDs             = (ID_VOICE *)              Allocator->Alloc((sizeof(ID_VOICE) * Size), HEAP_TAG_ENTITY_VOICE);
    Playbacks       = (TPL_PLAYBACK *)          Allocator->Alloc((sizeof(TPL_PLAYBACK) * Size), HEAP_TAG_ENTITY_VOICE);
    States          = (CMP_STATE *)             Allocator->Alloc((sizeof(CMP_STATE) * Size), HEAP_TAG_ENTITY_VOICE);
    Sources         = (HANDLE_SOURCE *)         Allocator->Alloc((sizeof(HANDLE_SOURCE) * Size), HEAP_TAG_ENTITY_VOICE);
    Types           = (TPL_TYPE *)              Allocator->Alloc((sizeof(TPL_TYPE) * Size), HEAP_TAG_ENTITY_VOICE);
    Amplitudes      = (CMP_FADE *)              Allocator->Alloc((sizeof(CMP_FADE) * Size), HEAP_TAG_ENTITY_VOICE);
    Crossfades      = (CMP_CROSSFADE *)         Allocator->Alloc((sizeof(CMP_CROSSFADE) * Size), HEAP_TAG_ENTITY_VOICE);    
    Pans            = (CMP_PAN *)               Allocator->Alloc((sizeof(CMP_PAN) * Size), HEAP_TAG_ENTITY_VOICE);
    Breakpoints     = (CMP_BREAKPOINT *)        Allocator->Alloc((sizeof(CMP_BREAKPOINT) * Size), HEAP_TAG_ENTITY_VOICE);
    ADSRs           = (CMP_ADSR *)              Allocator->Alloc((sizeof(CMP_ADSR) * Size), HEAP_TAG_ENTITY_VOICE);
    Filters         = (TPL_FILTER *)            Allocator->Alloc((sizeof(TPL_FILTER) * Size), HEAP_TAG_ENTITY_VOICE);
    Positions       = (CMP_POSITION *)          Allocator->Alloc((sizeof(CMP_POSITION) * Size), HEAP_TAG_ENTITY_VOICE);
    Distances       = (CMP_DISTANCE *)          Allocator->Alloc((sizeof(CMP_DISTANCE) * Size), HEAP_TAG_ENTITY_VOICE);
    Sines           = (CMP_CUDA_SINE *)         Allocator->Alloc((sizeof(CMP_CUDA_SINE) * Size), HEAP_TAG_ENTITY_VOICE);
    Bubbles         = (TPL_BUBBLES *)           Allocator->Alloc((sizeof(TPL_BUBBLES) * Size), HEAP_TAG_ENTITY_VOICE);
    //! This structs are over 4mb!
    // Grains          = (CMP_GRAIN *)             Allocator->Alloc((sizeof(CMP_GRAIN) * Size), HEAP_TAG_ENTITY_VOICE);    
    // FFTs            = (CMP_FFT *)               Allocator->Alloc((sizeof(CMP_FFT) * Size), HEAP_TAG_ENTITY_VOICE);
    Flags           = (i32 *)                   Allocator->Alloc((sizeof(i32) * Size), HEAP_TAG_ENTITY_VOICE);

    //Initialise
    Count = 0;
}

void ENTITY_VOICES::Destroy(MEMORY_ALLOCATOR *Allocator)
{
    Allocator->Free(0, HEAP_TAG_ENTITY_VOICE);
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

bool ENTITY_VOICES::Remove(ID_VOICE ID)
{
    for(ID_VOICE i = 0; i < Count; ++i)
    {
        if(ID == IDs[i])
        {
            //Free and reset
            Init(Count);

            //Decrement total count
            --Count;
            return true;
        }

        else
        {
            Fatal("Voices: Couldn't find entity! ID: %llu", ID);
        }
    }

    return false;
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
