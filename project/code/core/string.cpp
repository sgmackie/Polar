typedef char *STRING;

typedef struct STRING_HEADER 
{
	size_t len;
	size_t cap;
} STRING_HEADER;

#define StringHeaderBuild(String) ((STRING_HEADER *) String - 1)


STRING StringAlloc(void const *init_str, size_t len) {
	STRING str;
	STRING_HEADER *header;
	size_t header_size = sizeof(STRING_HEADER);
    void *ptr = malloc((header_size + len + 1));
	if (ptr == 0)
		return 0;
	if (!init_str)
		memset(ptr, 0, header_size + len + 1);

	str = (char *)ptr + header_size;
	header = StringHeaderBuild(str);
	header->len = len;
	header->cap = len;
	if (len && init_str)
		memcpy(str, init_str, len);
	str[len] = '\0';

	return str;
}

STRING StringCreate(char const *str = "") {
	size_t len = str ? strlen(str) : 0;
	return StringAlloc(str, len);
}

STRING StringCreateWithLength(char const *str, size_t len) {
	return StringAlloc(str, len);
}


void StringDestroy(STRING str) {
	if (str == 0)
		return;

    free((STRING_HEADER *)str - 1);
}
