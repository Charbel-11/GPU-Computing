#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <sys/time.h>

enum PrintColor { NONE, GREEN, RED, BLUE };

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

static void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

static void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

static void printElapsedTime(Timer timer, const char* s, enum PrintColor color = NONE) {
    float t = ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec) \
        + (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
		
	if (color == RED) { printf("\033[0;31m"); }
	else if (color == GREEN) { printf("\033[1;32m"); }
	else if (color == BLUE) { printf("\033[0;34m"); }
	
    printf("%s: %f s\n", s, t);
    
	if (color != NONE) { printf("\033[0m"); }
}

#endif