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
    f32 Random = ((f32) pcg32_random() / ((f32) UINT32_MAX));
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

void PiWrap(f32 &x)
{
    if(x < -PI32)
    {
        x += TWO_PI32;
    }

    else if (x > PI32)
    {
        x -= TWO_PI32;   
    }
}

f32 MiniMax(f32 x)
{
    PiWrap(x);

    return x * Series(x * x);
}


//Remez Exchange Polynomial with range [-1:1]
f32 polar_sin(f32 x)
{
    PiWrap(x);

    f32 u = -1.9227257e-4f;
    u = u * x + -3.402273e-8f;
    u = u * x + 8.3287542e-3f;
    u = u * x + 2.9187311e-8f;
    u = u * x + -1.666654e-1f;
    u = u * x + -5.7378907e-9f;
    u = u * x + 9.9999991e-1f;
    return u * x + 1.4656652e-10f;
}


//Remez Exchange Polynomial with range [-1:1]
f32 polar_cos(f32 x)
{
    PiWrap(x);

    f32 u = 8.7948712e-5f;
    u = u * x + -1.4322097e-3f;
    u = u * x + -1.4676678e-4f;
    u = u * x + 4.1787766e-2f;
    u = u * x + 7.3992365e-5f;
    u = u * x + -5.0005696e-1f;
    u = u * x + -1.118184e-5f;
    return u * x + 1.0000036f;
}




f32 VectorLength(VECTOR4D A)
{
    f32 Result = 0;
    f32 X = A.X * A.X;
    f32 Y = A.Y * A.Y;
    f32 Z = A.Z * A.Z;
    Result = sqrt((X + Y + Z));
    return Result;
}

VECTOR4D VectorSub(VECTOR4D A, VECTOR4D B)
{
    VECTOR4D Result = {};

    Result.X = A.X - B.X;
    Result.Y = A.Y - B.Y;
    Result.Z = A.Z - B.Z;
    Result.W = A.W - B.W;

    return Result;
}


void MatrixInverse(MATRIX_4x4 &Input, MATRIX_4x4 &Output)
{
    Output.A1 = Input.A1;
    Output.B2 = Input.B2;
    Output.C3 = Input.C3;
    Output.D4 = Input.D4;

    f32 Temp1 = Input.A2;
    f32 Temp2 = Input.B1;
    Output.A2 = Temp2;
    Output.B1 = Temp1;

    Temp1 = Input.A3;
    Temp2 = Input.C1;
    Output.A3 = Temp2;
    Output.C1 = Temp1;

    f32 Temp3 = Input.C2;
    Temp2 = Input.B3;
    Output.C2 = Temp2;
    Output.B3 = Temp3;

    Temp1 = -(Output.A1 * Input.D1 +
            Output.B1 * Input.D2 +
            Output.C1 * Input.D3);

    Temp2 = -(Output.A2 * Input.D1 +
            Output.B2 * Input.D2 +
            Output.C2 * Input.D3);

    Temp3 = -(Output.A3 * Input.D1 +
            Output.B3 * Input.D2 +
            Output.C3 * Input.D3);

    Output.D1 = Temp1;
    Output.D2 = Temp2;
    Output.D3 = Temp3;

    Output.A4 = 0.0f;
    Output.B4 = 0.0f;
    Output.C4 = 0.0f;
}

VECTOR4D ApplyMatrix(VECTOR4D Vector, MATRIX_4x4 Matrix)
{
    VECTOR4D Result = {};

    Result.X = (Vector.X * Matrix.A1) +
                (Vector.Y * Matrix.B1) +
                (Vector.Z * Matrix.C1) +
                (Vector.W * Matrix.D1);
    
    Result.Y = (Vector.X * Matrix.A2) +
                (Vector.Y * Matrix.B2) +
                (Vector.Z * Matrix.C2) +
                (Vector.W * Matrix.D2);
    
    Result.Z = (Vector.X * Matrix.A3) +
                (Vector.Y * Matrix.B3) +
                (Vector.Z * Matrix.C3) +
                (Vector.W * Matrix.D3);
    
    Result.W = (Vector.X * Matrix.A4) +
                (Vector.Y * Matrix.B4) +
                (Vector.Z * Matrix.C4) +
                (Vector.W * Matrix.D4);

    return Result;
}

VECTOR4D TransformToListener(VECTOR4D Source, VECTOR4D Listener)
{
    MATRIX_4x4 InvListener = {};
    InvListener.A2 = -1.0f;
    InvListener.B1 = 1.0f;
    InvListener.C3 = 1.0f;
    InvListener.D4 = 1.0f;

    InvListener.D1 = 0.0f;
    InvListener.D2 = 0.0f;
    InvListener.D3 = 0.0f;

    MatrixInverse(InvListener, InvListener);

    VECTOR4D PosInListenerSpace = VectorSub(Source, Listener);
    VECTOR4D RelativePostion = ApplyMatrix(PosInListenerSpace, InvListener);

    // printf("X: %f\tY: %f\tZ: %f\tW: %f\n", RelativePostion.X, RelativePostion.Y, RelativePostion.Z, RelativePostion.W);

    

    return RelativePostion;
}




#endif