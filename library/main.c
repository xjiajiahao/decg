#include <stdio.h>
#include "foo.h"

int main()
{
    int a = 1, b = 2, res;
    res = c_add(a, b);
    printf("This is a shared library test...  res = %d\n", res);
    return 0;
}
