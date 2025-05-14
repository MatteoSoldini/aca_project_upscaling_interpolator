lanczos: lanczos.c
	if not exist build mkdir build
	clang -Wall -Wextra -Ideps lanczos.c -o build/lanczos.exe