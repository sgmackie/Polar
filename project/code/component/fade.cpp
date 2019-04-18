void CMP_FADE::Init(f64 Amplitude)
{
    Current = Amplitude;
    Previous = Current;
    IsFading = false;
}