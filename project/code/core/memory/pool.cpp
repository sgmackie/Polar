

void MEMORY_POOL::FreeAll() 
{
	//Clear all chunks
	size_t ChunkCount = (Length / ChunkSize);
	for(size_t i = 0; i < ChunkCount; ++i) 
	{
		//Get node
		void *Pointer = &Data[i * ChunkSize];
		MEMORY_POOL_NODE *Node = (MEMORY_POOL_NODE *) Pointer;
		
		//Push node to free list
		Node->Next = Head;
		Head = Node;
	}

    Info("Pool: Freed all memory");
}

void MEMORY_POOL::Init(void *Block, size_t BlockLength, size_t BlockChunkSize, size_t BlockChunkAlignment)
{
	//Align backing block to the chunk alignment
	uintptr_t Start = (uintptr_t) Block;
	uintptr_t Alignment = AlignForwardPointer(Start, (uintptr_t) BlockChunkAlignment);
	BlockLength -= (size_t) (Alignment - Start);

	//Align chunk size to the chunk alignment
	BlockChunkSize = AlignForwardSize(BlockChunkSize, BlockChunkAlignment);

	//Assertions on input parameters
    Assert(BlockChunkSize >= sizeof(MEMORY_POOL_NODE), "Pool: Chunk size is too small! Size: %zu bytes", BlockChunkSize);
    Assert(BlockLength >= BlockChunkSize, "Pool: Backing block is smaller than the requested chunk size! Block length %zu bytes | Size: %zu bytes", BlockLength, BlockChunkSize);

	//Store parameters
	Data 		= (unsigned char *) Block;
	Length 		= BlockLength;
	ChunkSize 	= BlockChunkSize;
	Head 		= 0;

	//Create free list
	FreeAll();
}

void *MEMORY_POOL::Alloc()
{
	//Get last free node
	MEMORY_POOL_NODE *Node = Head;

	if(Node == 0)
	{
		Assert(0, "Pool: No free memory in the backing block!");
		return 0;
	}

	//Pop free node
	Head = Head->Next;

	//Clear to zero
    Info("Pool: Allocated chunk %zu bytes", ChunkSize);
	return memset(Node, 0, ChunkSize);
}

void MEMORY_POOL::Free(void *Pointer)
{
    //Set start and end points in the list
	MEMORY_POOL_NODE *Node = 0;
	void *Start = Data;
	void *End 	= &Data[Length];

    //Blank pointer - nothing to free
	if(Pointer == 0) 
	{
		Warning("Pool: Nothing to free");
		return;
	}

    //Pointer is not within the start and end bounds
	if(!(Start <= Pointer && Pointer < End)) 
	{
		Assert(0, "Size is out of bounds for the backing block!");
		return;
	}

	//Push free node
	Node = (MEMORY_POOL_NODE *) Pointer;
	Node->Next = Head;
	Head = Node;

    Info("Pool: Free");
}