
void CMP_RINGBUFFER::Create(MEMORY_ARENA *Arena, size_t Type, size_t InputCount, size_t Blocks)
{
    if(Blocks <= 0)
    {
        Blocks = RINGBUFFER_DEFAULT_BLOCK_COUNT;
        Info("Ringbuffer: Using default block count: %d", RINGBUFFER_DEFAULT_BLOCK_COUNT);
    }

    Count           = InputCount;
    TotalBlocks     = Blocks;    
    CurrentBlocks   = 0;
    ReadAddress     = 0;
    WriteAddress    = 0;

    Data = (i16 *)  Arena->Alloc(((Type * Count) * TotalBlocks), MEMORY_ARENA_ALIGNMENT);
}

void CMP_RINGBUFFER::Destroy()
{
    Count           = 0;
    TotalBlocks     = 0;    
    CurrentBlocks   = 0;
    ReadAddress     = 0;
    WriteAddress    = 0;
    Data            = 0;
}

i16 *CMP_RINGBUFFER::Write()
{
    return Data + (WriteAddress * Count);
}

bool CMP_RINGBUFFER::CanWrite()
{
    return CurrentBlocks != TotalBlocks;
}

void CMP_RINGBUFFER::FinishWrite()
{
    WriteAddress = ((WriteAddress + 1) % TotalBlocks);
    CurrentBlocks += 1;
}

i16 *CMP_RINGBUFFER::Read()
{
    return Data + (ReadAddress * Count);
}

bool CMP_RINGBUFFER::CanRead()
{
    return CurrentBlocks != 0;
}

void CMP_RINGBUFFER::FinishRead()
{
    ReadAddress = ((ReadAddress + 1) % TotalBlocks);
    CurrentBlocks -= 1;        
}    