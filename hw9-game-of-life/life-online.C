// Author: Roger Wang
// Modified Game of Life on CPU by Daniel Angel Jimenez

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BOARD_WIDTH 40
#define BOARD_HEIGHT 40

void createBoard(int board[][BOARD_HEIGHT]) {
    int i, j;

    for (i = 0; i < BOARD_WIDTH; i++)
        for (j = 0; j < BOARD_HEIGHT; j++)
            board[i][j] = rand() % 2;  // either 0 or 1
}

int xadd(int i, int a) {
    i += a;
    while (i < 0) i += BOARD_WIDTH;
    while (i >= BOARD_WIDTH) i -= BOARD_WIDTH;
    return i;
}

int yadd(int i, int a) {
    i += a;
    while (i < 0) i += BOARD_HEIGHT;
    while (i >= BOARD_HEIGHT) i -= BOARD_HEIGHT;
    return i;
}

int adjacentCells(int board[][BOARD_HEIGHT], int i, int j) {
    int count = 0;

    for (int k = -1; k <= 1; k++)
        for (int l = -1; l <= 1; l++)
            if (k || l)
                if (board[xadd(i, k)][yadd(j, l)]) count++;
    return count;
}

void play(int board[][BOARD_HEIGHT]) {
    int newboard[BOARD_WIDTH][BOARD_HEIGHT];
    int a;

    for (int i = 0; i < BOARD_WIDTH; i++) {
        for (int j = 0; j < BOARD_HEIGHT; j++) {
            int adj = adjacentCells(board, i, j);
            if (adj == 2) newboard[i][j] = board[i][j];
            if (adj == 3) newboard[i][j] = 1;
            if (adj < 2) newboard[i][j] = 0;
            if (adj > 3) newboard[i][j] = 0;
        }
    }

    for (int i = 0; i < BOARD_WIDTH; i++) {
        for (int j = 0; j < BOARD_HEIGHT; j++) {
            board[i][j] = newboard[i][j];
        }
    }
}

void showBoard(int board[][BOARD_HEIGHT]) {
    for (int j = 0; j < BOARD_HEIGHT; j++) {
        for (int i = 0; i < BOARD_WIDTH; i++) {
            printf("%s", board[i][j] ? "██" : "  ");
        }
        printf("\n");
    }
    for (int i = 0; i < BOARD_WIDTH - 1; i++)
        printf("_");
    printf("\n");
}

int main(int argc, char *argv[]) {
    int gameBoard[BOARD_WIDTH][BOARD_HEIGHT];

    createBoard(gameBoard);
    printf("%d", argc);

    while (1) {
        showBoard(gameBoard);
        play(gameBoard);
        usleep(1000 * 25);
    }
}