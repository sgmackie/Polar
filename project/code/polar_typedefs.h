//Redefining standard types to personal taste
#include <stdint.h>

//Macros
//Bug checking
#if defined(__clang__)
#include <assert.h>
#define Assert assert

#elif defined(_MSC_VER)
#define Assert(Expression) if(!(Expression)) {*(int *)0 = 0;}   //Standard assertion
#endif

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))  //Get count of elements in array

//Sizes
#define Kilobytes(Value) ((Value) * 1024LL) //long long forces this to be 64bit calculation
#define Megabytes(Value) (Kilobytes(Value) * 1024LL)
#define Gigabytes(Value) (Megabytes(Value) * 1024LL)
#define Terabytes(Value) (Gigabytes(Value) * 1024LL)

//Alignment
#define AlignPow2(Value, Alignment) ((Value + ((Alignment) - 1)) & ~((Value - Value) + (Alignment) - 1))    //Take the alignment value (power of 2) and round up by removing the bottom bits
#define Align4(Value)   ((Value + 3) & ~3)
#define Align8(Value)   ((Value + 7) & ~7)
#define Align16(Value)  ((Value + 15) & ~15)
#define Align32(Value)  ((Value + 31) & ~31)

//3 ways to use static
#define global_scope static       //Global access to variable
#define internal_scope static     //Variable local to source file only
#define local_scope static        //Variable persists after stepping out of scope (this should be avoided when possible)

//Types include longform and short form names
//Standard integers
//Unsigned
//8-bit
typedef uint8_t uint8;
typedef uint8_t u8;
//16-bit
typedef uint16_t uint16;
typedef uint16_t u16;
//32-bit
typedef uint32_t uint32;
typedef uint32_t u32;
//64-bit
typedef uint64_t uint64;
typedef uint64_t u64;

//Signed
//8-bit
typedef int8_t int8;
typedef int8_t i8;
//16-bit
typedef int16_t int16;
typedef int16_t i16;
//32-bit
typedef int32_t int32;
typedef int32_t i32;
//64-bit
typedef int64_t int64;
typedef int64_t i64;

//Floating points
//32-bit
typedef float float32;
typedef float f32;

//64-bit
typedef double float64;
typedef double f64;
