#ifndef polar_buffer_cpp
#define polar_buffer_cpp

POLAR_RINGBUFFER *polar_ringbuffer_Create(MEMORY_ARENA *Arena, u32 Samples, u32 Blocks)
{
    if(Blocks <= 0)
    {
        Blocks = RINGBUFFER_DEFAULT_BLOCK_COUNT;
    }

    POLAR_RINGBUFFER *Result = 0;
    Result = (POLAR_RINGBUFFER *) memory_arena_Push(Arena, Result, (sizeof (*Result)));
    
    Result->Samples = Samples;
    Result->TotalBlocks = Blocks;
    Result->Data = (f32 *) memory_arena_Push(Arena, Result->Data, ((sizeof (*Result->Data)) * (Result->Samples * Result->TotalBlocks)));
    
    Result->CurrentBlocks = 0;
    Result->ReadAddress = 0;
    Result->WriteAddress = 0;

    return Result;
}

void polar_ringbuffer_Destroy(MEMORY_ARENA *Arena, POLAR_RINGBUFFER *Buffer)
{
    if(Buffer)
    {
        memory_arena_Pull(Arena);

        Buffer->Samples = 0;
        Buffer->TotalBlocks = 0;    
        Buffer->CurrentBlocks = 0;
        Buffer->ReadAddress = 0;
        Buffer->WriteAddress = 0;
        Buffer = 0;
    }
}

f32 *polar_ringbuffer_WriteData(POLAR_RINGBUFFER *Buffer)
{
    return Buffer->Data + (Buffer->WriteAddress * Buffer->Samples);
}

bool polar_ringbuffer_WriteCheck(POLAR_RINGBUFFER *Buffer)
{
    return Buffer->CurrentBlocks != Buffer->TotalBlocks;
}

void polar_ringbuffer_WriteFinish(POLAR_RINGBUFFER *Buffer)
{
    Buffer->WriteAddress = ((Buffer->WriteAddress + 1) % Buffer->TotalBlocks);
    Buffer->CurrentBlocks += 1;
}

f32 *polar_ringbuffer_ReadData(POLAR_RINGBUFFER *Buffer)
{
    return Buffer->Data + (Buffer->ReadAddress * Buffer->Samples);
}

bool polar_ringbuffer_ReadCheck(POLAR_RINGBUFFER *Buffer)
{
    return Buffer->CurrentBlocks != 0;
}

void polar_ringbuffer_ReadFinish(POLAR_RINGBUFFER *Buffer)
{
    Buffer->ReadAddress = ((Buffer->ReadAddress + 1) % Buffer->TotalBlocks);
    Buffer->CurrentBlocks -= 1;
}


#endif