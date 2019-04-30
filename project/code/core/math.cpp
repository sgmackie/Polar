
#define CLAMP(x, a, b)    ((x) < (a) ? (a) : (x) > (b) ? (b) : (x))
#define MIN(a, b)         ((a) < (b) ? (a) : (b))
#define MAX(a, b)         ((a) > (b) ? (a) : (b))

f64 InterpLinear(f64 A, f64 B, f64 Delta)
{
    f64 Result = ((1.0 - Delta) * A + Delta * B);
    return Result;
}

f64 InterpLog(f64 A, f64 B, f64 Delta)
{
    f64 Result = ((log(Delta - 0.2) + A) * B);
    return Result;
}




f64 DecibelToLinear(f64 X)
{
    return (pow(10, (X / 20)));
}
#define DB(X) DecibelToLinear(X)


//https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
u32 UpperPowerOf2(u32 A)
{
    A--;

    A |= A >> 1;
    A |= A >> 2;
    A |= A >> 4;
    A |= A >> 8;
    A |= A >> 16;
    
    A++;
    
    return A;
}

u32 LowerPowerOf2(u32 A)
{
    u32 v = A; 

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    u32 x = v >> 1;

    return x;
}

u32 NearestPowerOf2(u32 n)
{
    u32 v = n; 

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++; // next power of 2

    u32 x = v >> 1; // previous power of 2

    return (v - n) > (n - x) ? x : v;
}

i16 FloatToInt16(f32 Input)
{
    i16 Result = 0;
    f32 Float = Input;
    Float = Float * 32768;

    if(Float > 32767)
    {
        Float = 32767;
    }
    
    if(Float < -32768) 
    {
        Float = -32768;
    }

    Result = (int16) Float;

	return Result;
}


f32 RandomFloat32(f32 Min, f32 Max) 
{    
    f32 Random      = ((f32) pcg32_random() / ((f32) UINT32_MAX));
    f32 Difference  = Max - Min;
    f32 Result      = Random * Difference;
    Result          += Min;
    
    return Result;
}


f64 RandomFloat64(f64 Min, f64 Max) 
{    
    f64 Random      = ((f64) pcg32_random() / ((f64) UINT32_MAX));
    f64 Difference  = Max - Min;
    f64 Result      = Random * Difference;
    Result          += Min;
    
    return Result;
}
