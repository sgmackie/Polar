

void CMP_BREAKPOINT::Init(size_t PointIndex, f64 InputValue, f64 InputTime)
{
    Index   = 0;
    if(PointIndex)
    {
        Points[PointIndex].Value = InputValue;
        Points[PointIndex].Time = InputTime;
    }
}

void CMP_BREAKPOINT::CreateFromArena(MEMORY_ARENA *Arena, size_t Type, size_t InputCount)
{
    Count   = 0;
    Points  = (CMP_BREAKPOINT_POINT *) Arena->Alloc((Type * InputCount), MEMORY_ARENA_ALIGNMENT);
}

void CMP_BREAKPOINT::CreateFromPool(MEMORY_POOL *Pool, size_t InputCount)
{
    Count   = 0;
    Points  = (CMP_BREAKPOINT_POINT *) Pool->Alloc();
}

void CMP_BREAKPOINT::Destroy()
{
    Index = 0;
    Count = 0;
    Points = 0;
}
