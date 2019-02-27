#ifndef polar_memory_cpp
#define polar_memory_cpp

//Chunks
MEMORY_ARENA_CHUNK *memory_arena_ChunkCreate(size_t Size) 
{
    size_t Query = ReturnMax(Size, ARENA_DEFAULT_CHUNK_SIZE);
    size_t AllocationSize = RoundToAlignmentBoundary(MaxAlignment, Query);
    size_t Alignment = RoundToAlignmentBoundary(MaxAlignment, offsetof(MEMORY_ARENA_CHUNK, Data));

#ifdef _WIN32    
    MEMORY_ARENA_CHUNK *Result = (MEMORY_ARENA_CHUNK *) VirtualAlloc(0, (Alignment + AllocationSize * (sizeof(Result->Data[0]))), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#elif __linux__
    MEMORY_ARENA_CHUNK *Result = (MEMORY_ARENA_CHUNK *) mmap(0, (Alignment + AllocationSize * (sizeof(Result->Data[0]))), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#else
    MEMORY_ARENA_CHUNK *Result = (MEMORY_ARENA_CHUNK *) aligned_alloc(MaxAlignment, (Alignment + AllocationSize * (sizeof(Result->Data[0]))));
#endif

    Result->CurrentSize = 0;
    Result->TotalSize = AllocationSize;
    Result->NextChunk = 0;
    
    return Result;
}

void memory_arena_ChunkDestroy(MEMORY_ARENA_CHUNK *Chunk) 
{
#ifdef _WIN32
    VirtualFree(Chunk, 0, MEM_RELEASE);
#elif __linux__
    size_t Alignment = RoundToAlignmentBoundary(MaxAlignment, offsetof(MEMORY_ARENA_CHUNK, Data));
    munmap(Chunk, (Alignment + Chunk->TotalSize * (sizeof(Chunk->Data[0]))));
#else
    free(Chunk);
#endif
}

//Arena
MEMORY_ARENA *memory_arena_Create(size_t Size)
{
    if(Size <= 0)
    {
        Size = ARENA_DEFAULT_CHUNK_SIZE;
    }

#ifdef _WIN32    
    MEMORY_ARENA *Result = (MEMORY_ARENA *) VirtualAlloc(0, (alignof(MEMORY_ARENA) * (sizeof(* Result))), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#elif __linux__
    MEMORY_ARENA *Result = (MEMORY_ARENA *) mmap(nullptr, (alignof(MEMORY_ARENA) * (sizeof(* Result))), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#else
    MEMORY_ARENA *Result = (MEMORY_ARENA *) aligned_alloc(alignof(MEMORY_ARENA), sizeof(* Result));
#endif

    MEMORY_ARENA_CHUNK *FirstChunk = memory_arena_ChunkCreate(Size);
    Result->UnalignedDataSize = 0;
    Result->CurrentSize = 0;
    Result->TotalSize = FirstChunk->TotalSize;
    Result->ChunkCount = 1;
    Result->FirstInList = FirstChunk;

    return Result;
}

void memory_arena_Destroy(MEMORY_ARENA *Arena) 
{
    if(Arena) 
    {
        Assert(Arena->FirstInList);
        
        for(MEMORY_ARENA_CHUNK *ChunkIndex = Arena->FirstInList; ChunkIndex; ChunkIndex = Arena->FirstInList) 
        {
            Arena->FirstInList = ChunkIndex->NextChunk;
            memory_arena_ChunkDestroy(ChunkIndex);
        }

#ifdef _WIN32
    VirtualFree(Arena, 0, MEM_RELEASE);
#elif __linux__
    munmap(Arena, (alignof(MEMORY_ARENA) * (sizeof(* Arena))));
#else
    free(Arena);
#endif
    }
}


void memory_arena_Reset(MEMORY_ARENA *Arena) 
{
    Assert(Arena);
    Assert(Arena->FirstInList);

    for(MEMORY_ARENA_CHUNK *ChunkIndex = Arena->FirstInList; ChunkIndex; ChunkIndex = ChunkIndex->NextChunk) 
    {
        ChunkIndex->CurrentSize = 0;
        memset(ChunkIndex->Data, 0, ChunkIndex->TotalSize);
    }

    Arena->UnalignedDataSize = 0;
    Arena->CurrentSize = 0;
}

void *memory_arena_Push(MEMORY_ARENA *Arena, void *Type, size_t Size) 
{
    Assert(Arena != 0);
    Assert(Arena->FirstInList != 0);

    size_t Alignment = alignof(decltype(Type));
    Size = (sizeof(decltype(Type)) * (Size));

    Assert(Alignment > 0);
    Assert(Size > 0);
    PowerOfTwoCheck(Alignment);
    AlignmentMultipleCheck(Alignment, Size);
    
    char *Data;
    size_t UnalignedDataSize;

    for(MEMORY_ARENA_CHUNK *ChunkIndex = Arena->FirstInList; ChunkIndex; ChunkIndex = ChunkIndex->NextChunk) 
    {
        Data = (ChunkIndex->Data + ChunkIndex->CurrentSize);
        UnalignedDataSize = ((RoundToAlignmentBoundary(Alignment, (uintptr_t) Data) - (uintptr_t) Data));

        if(((ChunkIndex->CurrentSize + UnalignedDataSize) + Size) <= ChunkIndex->TotalSize) 
        {
            Data += UnalignedDataSize;
            ChunkIndex->CurrentSize += UnalignedDataSize + Size;
            Arena->CurrentSize += Size;
            Arena->UnalignedDataSize += UnalignedDataSize;
            AlignmentMultipleCheck(Alignment, (uintptr_t) Data);
            
            return Data;
        }
    }

    MEMORY_ARENA_CHUNK *PushChunk = memory_arena_ChunkCreate(Size);
    AlignmentMultipleCheck(Alignment, (uintptr_t) PushChunk->Data);
    
    PushChunk->CurrentSize += Size;
    PushChunk->NextChunk = Arena->FirstInList;
    Arena->FirstInList = PushChunk;
    Arena->ChunkCount += 1;
    Arena->CurrentSize += Size;
    Arena->TotalSize += PushChunk->TotalSize;

    return PushChunk->Data;
}

void memory_arena_Pull(MEMORY_ARENA *Arena) 
{
    Assert(Arena);
    Assert(Arena->FirstInList);

    MEMORY_ARENA_CHUNK *CurrentHead = Arena->FirstInList;

    for(MEMORY_ARENA_CHUNK *ChunkIndex = CurrentHead->NextChunk; ChunkIndex; ChunkIndex = ChunkIndex->NextChunk)
    {
        if(ChunkIndex->CurrentSize == 0) 
        {
            CurrentHead->NextChunk = ChunkIndex->NextChunk;
            Arena->ChunkCount -= 1;
            Arena->TotalSize -= ChunkIndex->TotalSize;
            memory_arena_ChunkDestroy(ChunkIndex);
            ChunkIndex = CurrentHead;
        }
    }
}

void memory_arena_Print(MEMORY_ARENA *Arena)
{
    printf("Current Size: %zu\tTotal Size: %zu\tChunk Count: %zu\tUnaligned Data Size: %zu\n", Arena->CurrentSize, Arena->TotalSize, Arena->ChunkCount, Arena->UnalignedDataSize);
}


#endif