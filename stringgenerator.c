
#include<stdio.h>
#include<stdlib.h>

int INITIAL_WHILE = 1;
int END_WHILE = 12;

int main(){
    int i = INITIAL_WHILE;

    while (i <= END_WHILE){
        printf("data['timbre%.2d'][x] + ", i);

        i++;
    }
}
