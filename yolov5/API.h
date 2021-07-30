#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)  
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)  
#else
#define API  
#endif
#endif