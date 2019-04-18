//Functions
void MEMORY_ARENA::Init(void *Block, size_t Size)
{
    //Initialise
    Data            = (unsigned char *) Block;
    Length          = Size;
    PreviousOffset  = 0;
    CurrentOffset   = 0;        
}

void *MEMORY_ARENA::Alloc(size_t Size, size_t Alignment)
{
    //Find alignment offset
    uintptr_t CurrentPointer = (uintptr_t) Data + (uintptr_t) CurrentOffset;
    uintptr_t Offset = AlignForwardPointer(CurrentPointer, Alignment);
    Offset -= (uintptr_t) Data;

	//Check if there's enough space in the backing
	if((Offset + Size) <= Length) 
    {
		void *Allocation = &Data[Offset];
		PreviousOffset = Offset;
		CurrentOffset = (Offset + Size);

		//Set to zero
		memset(Allocation, 0, Size);
            
        Info("Arena: Allocated %zu bytes", Size);
    	return Allocation;
    }

    //Return 0 if there's not enough memory in the backing block
    Fatal("Arena: Not enough memory in backing block! Block length: %zu bytes | Requested size: %zu bytes", Length, Size);
    return 0;   
}

void *MEMORY_ARENA::Resize(void *OldAllocation, size_t OldSize, size_t NewSize, size_t Alignment)
{
    //Copy old memory and check alignment
    unsigned char *OldMemory = (unsigned char *) OldAllocation;    
    Assert(IsPowerOf2(Alignment), "Arena: Resize request isn't a power of 2! Alignment: %zu bytes", Alignment);

    //Previous allocation empty - normal allocation
    if(OldMemory == 0 || OldSize == 0) 
    {
        Info("Arena: Allocated %zu bytes", NewSize);
    	return Alloc(NewSize, Alignment);
    }

    //Perform resize
    else if(Data <= OldMemory && OldMemory < (Data + Length))
    {
        //Just use the old allocation
    	if((Data + PreviousOffset) == OldMemory) 
        {
    		CurrentOffset = PreviousOffset + NewSize;
            if(NewSize > OldSize) 
            {
    			//Set to zero
    			memset(&Data[CurrentOffset], 0, NewSize-OldSize);
    		}

    		return OldAllocation;
    	} 

        //Move pointer along to new allocation
        else 
        {
    		void *NewMemory = Alloc(NewSize, Alignment);
    		size_t CopySize = OldSize < NewSize ? OldSize : NewSize;
    		memmove(NewMemory, OldAllocation, CopySize);
	    		
            return NewMemory;
    	}
    }

    //Resize request is too big for the backing block
    else 
    {
        Assert(0, "Arena: Resize failed! Size is out of bounds for the backing block! Size: %zu bytes", NewSize);
    	return 0;
    }        
}

void MEMORY_ARENA::FreeAll()
{
    //Clear to 0
    PreviousOffset = 0;
    CurrentOffset = 0;
    Info("Arena: Freed all memory");
}



void MEMORY_ARENA_TEMPORARY::Begin(MEMORY_ARENA *Arena)
{
    //Copy over the offsets
	PreviousOffset  = Arena->PreviousOffset;
	CurrentOffset   = Arena->CurrentOffset;
    Info("Temporary Arena: Began block");
}

void MEMORY_ARENA_TEMPORARY::End()
{
    //Revert the offsets
	Arena->PreviousOffset   = PreviousOffset;
	Arena->CurrentOffset    = CurrentOffset;
    Info("Temporary Arena: Ended block");	
}