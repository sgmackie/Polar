

void CMP_BREAKPOINT::Init(size_t PointIndex, f64 InputValue, f64 InputTime)
{
    Index   = 0;
    if(PointIndex)
    {
        Points[PointIndex].Value = InputValue;
        Points[PointIndex].Time = InputTime;
    }
}

void CMP_BREAKPOINT::CreateFromPool(MEMORY_POOL *Pool, size_t InputCount)
{
    Count   = 0;
    Points  = (CMP_BREAKPOINT_POINT *) Pool->Retrieve();
}

void CMP_BREAKPOINT::FreeFromPool(MEMORY_POOL *Pool)
{
    Pool->Release(Points);
    Destroy();
}

void CMP_BREAKPOINT::Destroy()
{
    Index = 0;
    Count = 0;
    Points = 0;
}
