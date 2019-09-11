

void CMP_BUFFER::CreateFromPool(MEMORY_POOL *Pool, size_t InputCount)
{
    Count   = InputCount;
    Data    = (f32 *) Pool->Retrieve();
}

void CMP_BUFFER::FreeFromPool(MEMORY_POOL *Pool)
{
    Pool->Release(Data);
    Destroy();
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
