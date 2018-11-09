#include <stdio.h>
#include "foo.h"
int c_add(int arg_1, int arg_2) {
    printf("Hello, I am a shared library");
    return arg_1 + arg_2;
}
