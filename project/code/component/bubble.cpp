
void TPL_BUBBLES::Init(f64 SampleRate, u32 BubbleCount, f64 InputBubblesPerSec, f64 InputRadius, 
                        f64 InputAmplitude, f64 InputProbability, f64 InputRiseCutoff)
{
    Count               = UpperPowerOf2(BubbleCount);
    RadiusMinimum       = (3000 / (SampleRate / 2));
    RadiusMaximum       = InputRadius;
    BubblesPerSec       = InputBubblesPerSec;
    ProbabilityExponent = InputProbability;
    RiseCutoff          = InputRiseCutoff;
    Amplitude           = InputAmplitude;
    AmplitudeExponent   = 1.5;
    LambdaExponent      = 1.0;
    LambdaSum           = 0;
    Radii               = 0;
    Lambda              = 0;
    Generators          = 0;
}

void TPL_BUBBLES::Destroy()
{
    Generators = 0;
    Radii = 0;
    Lambda = 0;
}

void TPL_BUBBLES::CreateFromPool(MEMORY_POOL *GeneratorPool, MEMORY_POOL *RadiiPool, MEMORY_POOL *LambdaPool)
{
    Generators  = (CMP_BUBBLES_GENERATOR *) GeneratorPool->Retrieve();
    Radii       = (f64 *) RadiiPool->Retrieve();
    Lambda      = (f64 *) LambdaPool->Retrieve();
}

void TPL_BUBBLES::FreeFromPool(MEMORY_POOL *GeneratorPool, MEMORY_POOL *RadiiPool, MEMORY_POOL *LambdaPool)
{
    GeneratorPool->Release(Generators);
    RadiiPool->Release(Radii);
    LambdaPool->Release(Lambda);
    Destroy();
}