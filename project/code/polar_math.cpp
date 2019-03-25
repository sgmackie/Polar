#ifndef polar_math_cpp
#define polar_math_cpp

#ifndef MAX
#define MAX(A, B) (((A) > (B)) ? (A) : (B))
#endif

//Assertions
void PowerOfTwoCheck(size_t Value) 
{
    Assert((Value != 0) && ((Value & (~Value + 1)) == Value));
}

void AlignmentMultipleCheck(size_t Alignment, size_t Size) 
{
    Assert((Size % Alignment) == 0);
}

//Math
size_t ReturnMax(size_t A, size_t B) 
{
    return A > B ? A : B;
}

size_t ReturnMin(size_t A, size_t B) 
{
    return A < B ? A : B;
}

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

u64 UpperPowerOf2U64(u64 A)
{
    A--;

    A |= A >> 1;
    A |= A >> 2;
    A |= A >> 4;
    A |= A >> 8;
    A |= A >> 16;
    A |= A >> 32;
    
    A++;
    
    return A;
}



bool CheckFloat(f32 A, f32 Min, f32 Max)
{
    if(A >= Min && A <= Max)
    {
        return true;
    }

    return false;
}

f32 Int16ToFloat(i16 Input)
{
    f32 Result = 0;
    i16 Integer = Input;
    Result = ((f32) Integer) / (f32) 32768;
    if(Result > 1)
    {
        Result = 1;
    }

    if(Result < -1)
    {
        Result = -1;
    }

    return Result;
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



f32 math_RandomFloat32(f32 Min, f32 Max) 
{    
    f32 Random = ((f32)  pcg32_random() / ((f32) UINT32_MAX));
    f32 Difference = Max - Min;
    f32 Result = Random * Difference;
    Result += Min;
    
    return Result;
}


size_t RoundToAlignmentBoundary(size_t Alignment, size_t Size) 
{
    Assert(Alignment > 0);
    Assert(Size > 0);

    size_t Result = (((Size + Alignment - 1) / Alignment) * Alignment);
    
    return Result;
}


u64 math_HashGenerate(const char String[MAX_STRING_LENGTH])
{
    u32 InputLength = StringLength(String);
    
    if(InputLength > MAX_STRING_LENGTH)
    {
        printf("Hash\tERROR: Input string too long!\n");
        
        return 0;
    }

    u64 Result = XXH64(String, InputLength, 0);
    
    return Result;
}


f64 math_DecibelToLinear(f64 X)
{
    return (pow(10, (X / 20)));
}

f64 math_Truncate(f64 Input, u8 Precision)
{
    f64 Result = 0;
    if(Precision < 0 || Precision > 4)
    {
        return Result;
    }

    switch(Precision)
    {
        case 1:
        {
            i32 Truncation = Input * 10;
            Result = Truncation / 10.0;
            break;
        }
        case 2:
        {
            i32 Truncation = Input * 100;
            Result = Truncation / 100.0;
            break;
        }
        case 3:
        {
            i32 Truncation = Input * 1000;
            Result = Truncation / 1000.0;
            break;
        }
        case 4:
        {
            i32 Truncation = Input * 10000;
            Result = Truncation / 10000.0;
            break;
        }
        default:
        {
            i32 Truncation = Input * 100;
            Result = Truncation / 100.0;
            break; 
        }
    }

    return Result;
}


f32 Lerp(f32 From, f32 To, f32 Delta)
{
    return (1.0f - Delta) * From + Delta * To;
}

f32 Series(f32 x)
{
    f32 y = 0;
    
    f32 C[5] = {0.0000024609388329975758276f, -0.00019698112762578435577f, 0.0083298294559966612167f, -0.16666236485125293496f, 0.99999788400553332261f};

    
    y = C[0];
    y = y * x + C[1];
    y = y * x + C[2];
    y = y * x + C[3];
    y = y * x + C[4];

    return y;
}

f32 MiniMax(f32 x)
{
    if(x < -PI32)
    {
        x += TWO_PI32;
    }

    else if (x > PI32)
    {
        x -= TWO_PI32;   
    }

    return x * Series(x * x);
}


//Remex Exchange Polynomial with range [-1:1]
f64 polar_sin(f64 x)
{
    f64 u = -1.9227257000839916e-4;
    u = u * x + -3.4022729658798851e-8;
    u = u * x + 8.328754228903701e-3;
    u = u * x + 2.9187310895369112e-8;
    u = u * x + -1.6666540269736411e-1;
    u = u * x + -5.7378907443639691e-9;
    u = u * x + 9.9999990584636532e-1;
    return u * x + 1.4656652388792014e-10;
}

//Remex Exchange Polynomial with range [-1:1]
f64 polar_cos(f64 x)
{
    f64 u = 4.5207091507209189e-149;
    u = u * x + -1.3400704681371456e-3;
    u = u * x + -9.6128614515849353e-149;
    u = u * x + 4.1636329124213521e-2;
    u = u * x + 6.293016012824196e-149;
    u = u * x + -4.9999395278793666e-1;
    u = u * x + -1.1994354397101548e-149;
    return u * x + 9.9999981155164767e-1;
}


#endif