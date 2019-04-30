
void CMP_BUFFER::CreateFromArena(MEMORY_ARENA *Arena, size_t Type, size_t InputCount)
{
    Count   = InputCount;
    Data    = (f32 *) Arena->Alloc((Type * Count), MEMORY_ARENA_ALIGNMENT);
}

void CMP_BUFFER::CreateFromPool(MEMORY_POOL *Pool, size_t InputCount)
{
    Count   = InputCount;
    Data    = (f32 *) Pool->Alloc();
}

void CMP_BUFFER::Destroy()
{
    Count = 0;
    Data = 0;
}

f32 *CMP_BUFFER::Write()
{
    return Data;
}

f32 *CMP_BUFFER::Read()
{
    return Data;
}    
