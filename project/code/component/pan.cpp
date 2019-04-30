
void CMP_PAN::Init(i32 Mode, f64 Pan)
{
    Flag |= Mode;
    Amplitude = CLAMP(Pan, -1.0, 1.0);
}