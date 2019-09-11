
void CMP_GRAIN::Create()
{
    // Init
    Count       = 0;
    ListReader  = 0;

    //! Switch to pools
    Envelope    = (f32 *) malloc(sizeof(f32) * MAX_GRAIN_LENGTH);
    for(size_t i = 0; i < MAX_GRAINS; ++i)
    {
        Data[i] = (f32 *) malloc(sizeof(f32) * MAX_GRAIN_LENGTH);
    }
}

void CMP_GRAIN::Destroy()
{
    Count = 0;
    ListReader = 0;
    free(Envelope);
    
    //! Switch to pools
    for(size_t i = 0; i < MAX_GRAINS; ++i)
    {
        free(Data);
    }
}