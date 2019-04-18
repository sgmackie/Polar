bool IsPowerOf2(uintptr_t x) 
{
	return (x & (x-1)) == 0;
}

uintptr_t AlignForwardPointer(uintptr_t Pointer, size_t Alignment) 
{
	Assert(IsPowerOf2(Alignment), "Align: Requested size is not a power of two! Alignment: %zu bytes", Alignment);

    //Create a faster modulus - same as (P % A) but faster as 'A' is a power of two
	uintptr_t P = Pointer;
	uintptr_t A = (uintptr_t) Alignment;
	uintptr_t Modulo = P & (A - 1);

    //If pointer isn't aligned then add the alignment
	if(Modulo != 0) 
    {
		P += A - Modulo;
	}

	return P;
}

size_t AlignForwardSize(size_t Pointer, size_t Alignment) 
{
	Assert(IsPowerOf2(Alignment), "Align: Requested size is not a power of two! Alignment: %zu bytes", Alignment);

    //Create a faster modulus - same as (P % A) but faster as 'A' is a power of two
	size_t A = Alignment;
	size_t P = Pointer;
	size_t Modulo = P & (A - 1);
	
    //If pointer isn't aligned then add the alignment
	if(Modulo != 0) 
    {
		P += A - Modulo;
	}

	return P;
}