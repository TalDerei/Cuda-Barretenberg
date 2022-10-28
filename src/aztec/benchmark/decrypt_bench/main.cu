#include <chrono>
#include <iostream>
#include <sample.h>

using namespace std;

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main(int, char**)
{
    std::cout << "test" << std::endl;

    cuda_hello<<<1,1>>>(); 

    add(1,7);

    int sub = subtract(7, 1);
    cout << "sub is: " << sub << endl;
}
