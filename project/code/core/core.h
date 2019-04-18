#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "types.h"

#define PI32 3.14159265358979323846
#define TWO_PI32 (2.0 * PI32)



/* =============================================================================
 *
 *                                  Memory
 *
 * =============================================================================*/

//Sizes
#define Kilobytes(Value) ((Value) * 1024LL) //long long forces this to be 64bit calculation
#define Megabytes(Value) (Kilobytes(Value) * 1024LL)
#define Gigabytes(Value) (Megabytes(Value) * 1024LL)
#define Terabytes(Value) (Gigabytes(Value) * 1024LL)

//Alignments
#ifndef MEMORY_ARENA_ALIGNMENT
#define MEMORY_ARENA_ALIGNMENT 	(2 * sizeof(void *))
#endif

#ifndef MEMORY_POOL_ALIGNMENT
#define MEMORY_POOL_ALIGNMENT 	8
#endif



typedef struct MEMORY_ARENA 
{
    //Data
	unsigned char   *Data;              //Raw data
	size_t           Length;            //Length
	size_t           PreviousOffset;    //Location of the last data push
	size_t           CurrentOffset;     //Current offset in the data to push data from

    //Functions
    void Init(void *Block, size_t Size);
    void *Alloc(size_t Size, size_t Alignment);
    void *Resize(void *OldAllocation, size_t OldSize, size_t NewSize, size_t Alignment);
    void FreeAll();

} MEMORY_ARENA;



/* =============================================================================
 *
 *                                  TEMPORARY_ARENA
 *
 * =============================================================================*/
/*/// ### Temporary Arena
///
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~c
/// typedef struct MEMORY_ARENA_TEMPORARY
/// {
///     //Data
/// 	MEMORY_ARENA    *Arena;
/// 	size_t          PreviousOffset;
/// 	size_t          CurrentOffset;
///
///     //Functions
///     void Begin(MEMORY_ARENA *a)
///     void End()
/// } MEMORY_ARENA_TEMPORARY;
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
/// #### Reference
/// Function            | Description
/// --------------------|-------------------------------------------------------
/// __Begin__           | Start the temporary arena
/// __End__             | End the temporary arena
///
/// #### MEMORY_ARENA_TEMPORARY::Begin
/// Start the temporary arena
///
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~c
/// void Begin(MEMORY_ARENA *a)
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
/// Parameter   | Description
/// ------------|-----------------------------------------------------------
/// __a__       | A parent arena
///
/// #### MEMORY_ARENA_TEMPORARY::End
/// End the temporary arena
///
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~c
/// void End()
/// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
///
*/
typedef struct MEMORY_ARENA_TEMPORARY
{
    //Data
	MEMORY_ARENA *Arena;
	size_t PreviousOffset;
	size_t CurrentOffset;

    //Functions
    void Begin(MEMORY_ARENA *Arena);
    void End();

} MEMORY_ARENA_TEMPORARY;


/* =============================================================================
 *
 *                                  	POOL
 *
 * =============================================================================*/
/*/// ### Pool
///	Memory pools are used to allocate chunks of data of the same size, for example
/// a block of strings all 128 characters in length
///
*/
typedef struct MEMORY_POOL_NODE
{
	MEMORY_POOL_NODE 	*Next;
} MEMORY_POOL_NODE;

typedef struct MEMORY_POOL
{
    //Data
	unsigned char   	*Data;
	size_t           	Length;
	size_t           	ChunkSize;
	MEMORY_POOL_NODE 	*Head;

	//Functions
	void Init(void *Block, size_t BlockLength, size_t ChunkSize, size_t ChunkAlignment);
	void *Alloc();
	void Free(void *Pointer);
	void FreeAll();

} MEMORY_POOL;




/* =============================================================================
 *
 *                                  String
 *
 * =============================================================================*/



/* =============================================================================
 *
 *                                  Log
 *
 * =============================================================================*/




static const char *LOG_LEVELS[] = {"TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};
typedef enum FLAG_LOG_LEVELS {LOG_TRACE, LOG_DEBUG, LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_FATAL} FLAG_LOG_LEVELS;

typedef struct LOGGER
{
    //Data
    FILE    *File;
    void    *Data;
    bool    IsLocked;
    u8      Level;
    bool    IsQuiet;

    //Functions
    void Init(FILE *InputFile, u8 InputLevel, bool EnableQuiet);
    void *Lock();
    void Unlock(void *InputData);
    void Log(u8 InputLevel, const char *SourceFile, u64 SourceLine, const char *Format, ...);

} LOGGER;

//Macros
#define Trace(...)      Logger.Log(LOG_TRACE,    __FILE__, __LINE__, __VA_ARGS__)  
#define Debug(...)      Logger.Log(LOG_DEBUG,    __FILE__, __LINE__, __VA_ARGS__)  
#define Info(...)       Logger.Log(LOG_INFO,     __FILE__, __LINE__, __VA_ARGS__)  
#define Warning(...)    Logger.Log(LOG_WARNING,  __FILE__, __LINE__, __VA_ARGS__)  
#define Error(...)      Logger.Log(LOG_ERROR,    __FILE__, __LINE__, __VA_ARGS__)  
#define Fatal(...)      Logger.Log(LOG_FATAL,    __FILE__, __LINE__, __VA_ARGS__)  
