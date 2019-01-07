#ifndef polar_memory_cpp
#define polar_memory_cpp

//Initialise zero with a base address
void polar_memory_ArenaInitialise(POLAR_MEMORY_ARENA *Arena, size_t Size, void *BaseAddress)
{
    Arena->Size = Size;
    Arena->BaseAddress = (uint8 *)BaseAddress;
    Arena->UsedSpace = 0;
    Arena->TemporaryArenaCount = 0;
}

//Create temporary arena from a given parent
POLAR_MEMORY_TEMPORARY polar_memory_TemporaryArenaCreate(POLAR_MEMORY_ARENA *Parent)
{
    POLAR_MEMORY_TEMPORARY Arena;

    Arena.TemporaryArena = Parent;
    Arena.UsedSpace = Parent->UsedSpace;

    ++Parent->TemporaryArenaCount;

    return Arena;
}

//Remove arena from temporary memory space
void polar_memory_TemporaryArenaDestroy(POLAR_MEMORY_TEMPORARY TemporaryMemory)
{
    POLAR_MEMORY_ARENA *Arena = TemporaryMemory.TemporaryArena;
    Assert(Arena->UsedSpace >= TemporaryMemory.UsedSpace);

    Arena->UsedSpace = TemporaryMemory.UsedSpace;
    Assert(Arena->TemporaryArenaCount > 0);
    
    --Arena->TemporaryArenaCount;
}

//Find the offset size when pushing data onto an arena
size_t polar_memory_AlignmentOffsetGet(POLAR_MEMORY_ARENA *Arena, size_t Alignment)
{
    size_t Offset = 0;
    
    size_t Address = (size_t) Arena->BaseAddress + Arena->UsedSpace;
    size_t AlignmentMask = Alignment - 1;
    
    if(Address & AlignmentMask)
    {
        Offset = Alignment - (Address & AlignmentMask);
    }

    return Offset;
}

//From a given type (and size), push to the arena with a byte alignment
void *polar_PushSize(POLAR_MEMORY_ARENA *Arena, size_t SizeInitial, size_t Alignment)
{
    size_t Size = SizeInitial;
        
    size_t AlignmentOffset = polar_memory_AlignmentOffsetGet(Arena, Alignment);
    Size += AlignmentOffset;
    
    Assert((Arena->UsedSpace + Size) <= Arena->Size);

    void *Data = Arena->BaseAddress + Arena->UsedSpace + AlignmentOffset;
    Arena->UsedSpace += Size;

    Assert(Size >= SizeInitial);
    
    return Data;
}

#endif